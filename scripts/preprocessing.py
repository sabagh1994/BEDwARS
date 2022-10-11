import numpy as np
import pandas as pd


def converter(mixtures, signatures_ref, transformation, normalization):
    """
    Reference cell type signatures and mixtures are transformed and normalized.

    :param mixtures: (np.ndarray) bulk expression profiles
    :param signatures_ref: (np.ndarray) reference cell type signatures
    :param transformation: (str) "log", None
    :param normalization: (str) "mean", "standard"
    :return: (tuple) transformed and normalized bulk samples and reference cell type signatures
    """
    mix_mean = None
    if transformation == "log":
        signatures_ref = np.log1p(signatures_ref)
        mixtures = np.log1p(mixtures)
    if normalization == "mean":
        signatures_ref = signatures_ref - np.mean(signatures_ref, axis=0)
        mix_mean = np.exp(np.mean(mixtures, axis=0))
        mixtures = mixtures - np.mean(mixtures, axis=0)
        if transformation == "log":
            mixtures = np.exp(mixtures)
    elif normalization == "standard":
        signatures_ref = signatures_ref - np.mean(signatures_ref, axis=0)
        signatures_ref = signatures_ref/(np.var(signatures_ref, axis=0)**0.5)
        mixtures = mixtures - np.mean(mixtures, axis=0)
        mixtures = mixtures/(np.var(mixtures, axis=0)**0.5)
        if transformation == "log":
            mixtures = np.exp(mixtures)
    return mixtures, signatures_ref, mix_mean


def get_markers(marker_dict, mixtures, signatures_ref, signatures_org):
    """
    subset the genes to the markers
    currently works reliably if marker_type is set to "all" in marker_dict

    :param marker_dict: (dict) {'marker_type': None, 'marker_dir': None, "naive_marker_FC": 2.}
    :param mixtures: (pd.DataFrame) bulk expression profiles
    :param signatures_ref: (pd.DataFrame) reference signatures
    :param signatures_org: (pd.DataFrame) true signatures
    :return: subset mixtures, reference and true signatures to markers
    """
    # if markers then include them as well (intersect with markers)
    if marker_dict:
        marker_type = marker_dict.get("marker_type", "all")
        if marker_type == "provided":
            marker_dir = marker_dict.get("marker_dir", None)
            if marker_dir:
                markers = list(pd.read_csv(marker_dir, sep="\t")[0])
                ensid = list(mixtures.columns)[0]
                mixtures = mixtures[mixtures[ensid].isin(markers)]
                signatures_org = signatures_org[signatures_org[ensid].isin(markers)] \
                    if not(signatures_org is None) else None
                signatures_ref = signatures_ref[signatures_ref[ensid].isin(markers)]
            # in case no marker directory is found all genes are included
            ensid= list(mixtures.iloc[:, 0])
            mixtures = np.array(mixtures.iloc[:, 1:])
            signatures_org = np.array(signatures_org.iloc[:, 1:]) if not(signatures_org is None) else None
            signatures_ref = np.array(signatures_ref.iloc[:, 1:])

        elif marker_type == "naive":
            ensid = None  # should be adjusted later
            mixtures = np.array(mixtures.iloc[:, 1:])
            signatures_ref = np.array(signatures_ref.iloc[:, 1:])
            signatures_org = np.array(signatures_org.iloc[:, 1:]) if not(signatures_org is None) else None
            FC = marker_dict.get("naive_marker_FC", 2.)

            no_marker_genes = []
            for gene_idx in range(signatures_ref.shape[0]):
                gene_expr = signatures_ref[gene_idx, :]
                max_expr = np.sort(gene_expr)[-1]
                max_next = np.sort(gene_expr)[-2]
                if not(max_expr >= FC*max_next):
                    no_marker_genes.append(gene_idx)
            no_marker_genes = list(set(no_marker_genes))
            signatures_ref = np.delete(signatures_ref, no_marker_genes, axis=0)
            signatures_org = np.delete(signatures_org, no_marker_genes, axis=0) if not(signatures_org is None) else None
            mixtures = np.delete(mixtures, no_marker_genes, axis=0)

        elif marker_type == "all":
            ensid = list(mixtures.iloc[:, 0])
            mixtures = np.array(mixtures.iloc[:, 1:])
            signatures_ref = np.array(signatures_ref.iloc[:, 1:])
            signatures_org = np.array(signatures_org.iloc[:, 1:]) if not(signatures_org is None) else None
        else:
            print(f"invalid marker_type {marker_type}")
            raise NotImplementedError
        return mixtures, signatures_ref, signatures_org, ensid


def preprocessor(ref_sig_dir="./", mix_dir="./", org_sig_dir=None, org_prop_dir=None,
                 marker_dict=None, transformation="log", normalization="mean"):
    """
    1. Sort the columns (cell types) of signature(s) and rows of the original proportions if provided
    2. Take the common genes among signatures and bulk mixtures. Subset to marker genes
    3. Transform and normalize reference signatures and bulk mixtures

    :param ref_sig_dir: (str) Path to the reference signature
    :param mix_dir: (str) Path to the bulk expression profiles
    :param org_sig_dir: (str) Path to the true signature if available,
                        otherwise should be set to the path to the reference signature
                        (for method development purpose only)
    :param org_prop_dir: (str) Path to true proportions if available (for method development purpose only)
    :param marker_dict: (dict) {'marker_type': None, 'marker_dir': None, "naive_marker_FC": 2.}
                        marker_type could be "all", "naive", or "provided". If it is set to "provided"
                        then marker_dir should be specified otherwise all genes will be used. If it is
                        set to "naive" then naive marker selection is performed. FC needed for naive
                        marker selection can be passed by "naive_marker_FC".
                        if "all" is used for marker_type all the genes are used
                        (marker_type = "all" was used for all the experiments in the paper)
    :param transformation: (str) "log", None -- transformation on the signature and bulk expression values
                           "log" used for all the experiments in the paper
    :param normalization: (str) "mean", "standard" -- normalization on each bulk sample and reference cell type signature
                          "mean" used for all the experiments in the paper
    :return: (tuple) processed bulk profiles, reference/true signatures, proportions, cell type names and ensids
    """

    # load the mixtures and reference signature
    mixtures = pd.read_csv(mix_dir, sep='\t')  # G*N
    signatures_ref = pd.read_csv(ref_sig_dir, sep='\t')  # G*C
    signatures_org = pd.read_csv(org_sig_dir, sep='\t') \
        if org_sig_dir else None  # G*C # load the original signatures if provided
    props = pd.read_csv(org_prop_dir, sep='\t') if org_prop_dir else None  # C*N # load true proportions if provided

    # Assuming the first column has ENSIDs
    # sort the columns of mixture and signatures and the rows of proportions
    mix_names = list(mixtures.columns)[1:]
    ct_names = list(signatures_ref.columns)[1:]
    ensid = list(signatures_ref.columns)[0]
    signatures_ref = signatures_ref[list(signatures_ref.columns)]
    signatures_org = signatures_org[list(signatures_ref.columns)] if org_sig_dir else None
    props = props.reindex(ct_names) if org_prop_dir else None  # sort the rows

    # taking the common genes and sorting them
    com_ens = set(signatures_org[ensid]) & set(signatures_ref[ensid]) if org_sig_dir else set(signatures_ref[ensid])
    com_ens = set(com_ens) & set(mixtures[ensid])

    signatures_ref = signatures_ref[signatures_ref[ensid].isin(com_ens)]
    signatures_org = signatures_org[signatures_org[ensid].isin(com_ens)] if org_sig_dir else None
    mixtures = mixtures[mixtures[ensid].isin(com_ens)]

    # sorting genes by ensID
    mixtures.sort_values(by=[ensid], inplace=True)
    signatures_ref.sort_values(by=[ensid], inplace=True)
    signatures_org.sort_values(by=[ensid], inplace=True) if org_sig_dir else None

    mixtures, signatures_ref, signatures_org, ensid = get_markers(marker_dict, mixtures, signatures_ref, signatures_org)
    mixtures_org = np.copy(mixtures)
    # in log transformation returned ref sigs are in log space
    mixtures, signatures_ref, mix_mean = converter(mixtures, signatures_ref, transformation, normalization)
    props = np.array(props) if not(props is None) else None  # C*N

    return mixtures, signatures_ref, signatures_org, props, ct_names, ensid, mix_mean, mixtures_org, mix_names


if __name__ == '__main__':
    mix_dir = "/shared-mounts/sinhas/Saba/ANOVA/Deconv/MH_largeRuns/recent_benchamrk/EMTAB/T_d"
    ref_sig_dir = "/shared-mounts/sinhas/Saba/ANOVA/Deconv/MH_largeRuns/recent_benchamrk/Baron/C"
    org_sig_dir = "/shared-mounts/sinhas/Saba/ANOVA/Deconv/MH_largeRuns/recent_benchamrk/EMTAB/C_test_d"
    org_prop_dir = "/shared-mounts/sinhas/Saba/ANOVA/Deconv/MH_largeRuns/recent_benchamrk/EMTAB/P_d"
    marker_dict = {"marker_type": "all", "marker_dir": None, "naive_marker_FC": 2.}
    tup = preprocessor(ref_sig_dir=ref_sig_dir, mix_dir=mix_dir,
                       org_sig_dir=org_sig_dir, org_prop_dir=org_prop_dir,
                       marker_dict=marker_dict, transformation="log", normalization="mean")
    mixtures, signatures_ref, signatures_org, props, ct_names, ensid = tup
