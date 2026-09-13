#!/bin/bash
# Build the benchmark-oriented training split, the one ExpandDiff-B is trained
# on (`scenehdr_c95_512` in the paper): a rough match to how the SI-HDR
# benchmark constructs its own inputs.
#
# WHAT SI-HDR DOES (their paper, Sec. 3.2, Eq. 1):
#     y = q( min{1, g(e*x) + n} ),   n ~ N(0, alpha*x + beta)
#   e  exposure set to RETAIN 95% (clip_95) or 97% (clip_97) of HDR pixels
#   g  a camera response drawn from 85 curves clustered from the Grossberg &
#      Nayar (2003) DoRF database
#   n  signal-dependent Gaussian noise
#   q  quantisation to 8 bits
#
# WHAT THIS BUILD DOES, and where it departs -- deliberately rough:
#   e  -> percentile clip at U[3,7]% of pixels blown, spanning clip_97 (3%) and
#         clip_95 (5%). The same effect under a different parameterisation.
#   g  -> a 2-parameter analytic response, beta~N(0.6,0.1), gamma~N(0.9,0.1),
#         randomised per image, rather than the 85 DoRF curves. The model
#         therefore learns robustness to an unknown response, not to one curve.
#   n  -> NOT MODELLED. This is the largest remaining gap.
#   q  -> an 8-bit round trip, via --capture_model lediff.
#
# Nothing here tells the model which response curve a test image used, and the
# capture model inverts sRGB rather than the true curve on the way back -- the
# same inverse a deployed model applies to an unknown camera.
#
# A wrapper: run_scenehdr_pct_build.sh does the work and takes these settings
# from the environment. It refuses to overwrite an existing split.

set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)

export PCT_HIGH="3 7"        # blow U[3,7]% -- clip_97 is 3%, clip_95 is 5%
export PCT_LOW="0 0"         # crush NOTHING: SI-HDR clip_95/97 are 0.00% crushed
export CAPTURE=lediff        # randomised response curve + 8-bit round trip
export CLIP_REPEATS=3        # 39,768 pairs, matching the percentile split
export EXPOSURES=e0
export VIEWS=5
export OUT=${OUT:-$ROOT/data/scenehdr_c95_512}

echo "scenehdr_c95_512: high=U[$PCT_HIGH]%  low=U[$PCT_LOW]%  capture=$CAPTURE  repeats=$CLIP_REPEATS"
exec bash "$HERE/run_scenehdr_pct_build.sh"
