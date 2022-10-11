#!/bin/bash 


# Usage: download input files (signatures, proportions, and pseudo-bulk samples) from google drive

cd input
MIXFILEID="1C0FiGaaW8EtsTmWF1gNx6NChKMDjRyKg"
SIGFILEID="1ISq70T6ik1h2WEW34vnvkXHVzEEgsegU"
PROPFILEID="1kdAvL-Iff57-eEG_JequxqjRSFqe1xOw"
gdown $MIXFILEID
gdown $SIGFILEID
gdown $PROPFILEID

# sanity check
if ! md5sum -c input.md5; then echo "corrupted files"; fi


tar -xf mixtures.tar
tar -xf signatures.tar
tar -xf proportions.tar

rm *.tar
