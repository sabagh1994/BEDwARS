#!/bin/bash 


# Usage: download config files used for pancreas, brain (benchmarking) and dpd deficiency from google drive

cd configs
BRAINFILEID="12kuAsl6gjed2cyggZGPFcq07qTMkoQYi"
DPDFILEID="1Bwm4s82BL4W0GkI-pEOnSXrOWBhBFTuo"
PANCFILEID="1tTzj8aTnAS1jJeV_sExMT06AmZeVP7ac"
gdown $BRAINFILEID
gdown $DPDFILEID
gdown $PANCFILEID

# sanity check
if ! md5sum -c configs.md5; then echo "corrupted files"; fi


tar -xf brain.tar
tar -xf dpd_brain.tar
tar -xf pancreas.tar

rm *.tar
