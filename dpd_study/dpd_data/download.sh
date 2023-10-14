#!/bin/bash 


# Usage: download files required for preprocessing the dpd data, generating signatures and pseudo bulk samples for dpd expression decononvolution

DPDID="1hRqeDH0auxyZxz-K7UYAggNzdkt2SjvM"
gdown $DPDID

# sanity check
if ! md5sum -c dpd_data.md5; then echo "corrupted files"; fi

tar -xzvf dpd_data.tar.gz

rm *.tar.gz
