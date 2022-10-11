import torch


def argsort_chains(sort_criterion, model, X, point=None,
                   is_transformed=True, extra_dict=None, torch_sum=None):
    """
    Sort the chains based on the croterion provided
    :param sort_criterion: (str) The sorting criterion e.g., "LL", "posterior", "MSE_marker"
                            "MSE_marker" was used for the paper experiments. The other criteria should
                            be tested so please do not use them
    :param model: (Model.Model) The model object
    :param X: (torch.Tensor) The bulk expression profiles
    :param point: (dict) The point at which the chains are evaluated, i.e. the criterion is computed
    :param is_transformed: (boolean) Whether the point provided is transformed or not
        for the variables that require transformation
    :param extra_dict: (dict) Contains the attributes needed for the calculation of a criterion
                        e.g., in "MSE_marker" the fold change for the marker selection should be
                        specified via extra_dict.
    :param torch_sum: Function to perform summation
    :return: (tuple) The indices of the soted chains and the criterion value computed for the chains
                     which could be a list or Torch.tensor
    """

    if extra_dict is None:
        extra_dict = dict()

    if (point == {}) or (point is None):
        # use the current model variable values
        print("getting the current variables values to sort the chains")
        point = dict()
        is_transformed = True
        for var in model.vars:
            point[var.name] = var.value

    if sort_criterion in ['LL', 'posterior']:
        # for LL and posterior computation the point should be transformed
        if not is_transformed:
            point_trans = {}
            for var_name, var_value in point.items():
                point_trans[var_name] = var_value
                if var_name != 'B':
                    var_value_trans = model.__dict__[var_name].distribution.transform.forward(var_value)
                    point_trans[var_name] = var_value_trans
            point = point_trans
    elif sort_criterion in ['MSE_marker', 'MSE']:
        # if the point is transformed then backward should be performed
        if is_transformed:
            point_untrans = {}
            for var in model.vars:
                var_untrans = var.backward_var(point[var.name])
                point_untrans[var.name] = var_untrans
            point = point_untrans

    criterion = None
    if sort_criterion == 'LL':
        # the point as input to forward should be transformed
        mu_x = model.forward(point)  # point should be transformed
        sd_x = model.sigma_x.backward_var(point['sigma_x'])
        LL = model.LL(observed=X, mu=mu_x, sd=sd_x)
        sort_inds = torch.argsort(LL, dim=0, descending=True).reshape(-1)  # output is a tensor
        criterion = LL
    elif sort_criterion == 'posterior':
        # the point should be transformed
        posterior = model.logp(observed=X, point=point, torch_sum=torch_sum)
        sort_inds = torch.argsort(posterior, dim=0, descending=True).reshape(-1)  # output is a tensor
        criterion = posterior
    elif sort_criterion == 'MSE_marker':
        criterion = []
        FCs = extra_dict.get('FCs', [4., 5.])
        sort_inds = []
        for FC in FCs:
            marker_stats = get_marker_stats(model, X, point, FC=FC)
            X_MSE_marker = marker_stats['X_MSE_marker']
            sort_ind = torch.argsort(X_MSE_marker, dim=0, descending=False).reshape(-1)  # output is a tensor
            sort_inds.append(sort_ind)
            criterion.append(X_MSE_marker)
    else:
        raise ValueError(f"sort_criterion {sort_criterion} not available")
    return sort_inds, criterion


def get_marker_genes(sig_ref, FC=5.):
    """
    Identify the marker genes
    :param sig_ref: (torch.Tensor) Reference cell type signatures
    :param FC: (float) Fold change to pick the marker genes
    :return marker_genes: (list) Marker genes returned in a list
    """
    # implement in torch
    marker_genes = []
    for gene_idx in range(sig_ref.shape[0]):
        gene_expr = sig_ref[gene_idx, :]
        sorted_gene_expr = torch.abs(torch.sort(gene_expr)[0])
        max_expr, max_next = sorted_gene_expr[-1], sorted_gene_expr[-2]
        if max_expr >= FC*max_next:  # 1.5 for MM and 2 for IP
            marker_genes.append(gene_idx)
    marker_genes = list(set(marker_genes))
    print(f"count of markers is {len(marker_genes)}")
    return marker_genes


def get_marker_stats(model, X, point, FC=5.):
    """
    Compute the statistic of the model restricted to the marker genes
    :param model: (Model.Model) The model object
    :param X: (torch.Tensor) Bulk expression profiles
    :param point: (dict) The point at which the statistic is reported
    :param FC: (float) Fold change used for the identification of marker genes
    :return marker_stats: (dict) Only contains the marker MSE for the estimated
                          and true bulk expression profiles
    """

    sig_ref = model.sig_ref
    W, B = point['W'], point['B']
    sigma_b = point['sigma_b']

    if not model.fixed_sigs:
        u = sig_ref
        u = u.reshape((*model.dim_ext, model.G, model.C))  # adding the chain dimension
        cv = u * (sigma_b**2)
        cv = torch.abs(cv)
        cv = torch.sqrt(cv)
        S = u + (B*cv)
        if model.transformation == 'log':
            S = torch.exp(S)
    else:
        S = sig_ref   # .detach().clone() not sure if this is needed
        S = S.reshape((*model.dim_ext, model.G, model.C))  # (1, G, C)

    sig_ref = sig_ref.reshape((model.G, model.C))
    marker_genes = get_marker_genes(sig_ref, FC=FC)
    S_marker, X_marker = S[:, marker_genes, :], X[marker_genes, :]
    X_marker_hat = S_marker@W
    X_MSE_marker = torch.mean((X_marker - X_marker_hat)**2, dim=[-2, -1], keepdim=True)
    marker_stats = dict(X_MSE_marker=X_MSE_marker)  # later more can be added
    return marker_stats

