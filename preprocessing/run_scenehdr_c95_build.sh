#!/bin/bash
# Build the training split ExpandDiff-B is trained on, into OUT (default
# data/scenehdr_c95_512): a rough match to how the SI-HDR benchmark constructs
# its own inputs.
#
# SI-HDR's input model (their paper, Sec. 3.2, Eq. 1):
#     y = q( min{1, g(e*x) + n} ),   n ~ N(0, alpha*x + beta)
# with exposure e, camera response g, noise n and 8-bit quantisation q. This
# build realises each term as:
#   e  -> percentile clip at U[3,7]% of pixels blown.
#   g  -> a 2-parameter analytic response, beta~N(0.6,0.1), gamma~N(0.9,0.1),
#         randomised per image.
#   n  -> not modelled.
#   q  -> an 8-bit round trip, via --capture_model lediff.
# The capture model inverts sRGB rather than the true curve on the way back.
#
# A wrapper: run_scenehdr_pct_build.sh does the work and takes these settings
# from the environment. It refuses to overwrite an existing split.

set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)

export PCT_HIGH="3 7"        # blow U[3,7]% of pixels
export PCT_LOW="0 0"         # crush nothing
export CAPTURE=lediff        # randomised response curve + 8-bit round trip
export CLIP_REPEATS=3        # as in the percentile split
export EXPOSURES=e0
export VIEWS=5
export OUT=${OUT:-$ROOT/data/scenehdr_c95_512}

echo "scenehdr_c95_512: high=U[$PCT_HIGH]%  low=U[$PCT_LOW]%  capture=$CAPTURE  repeats=$CLIP_REPEATS"
exec bash "$HERE/run_scenehdr_pct_build.sh"
