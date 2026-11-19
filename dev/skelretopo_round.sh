#!/bin/zsh
# One full skeleton-retopo round for a character with a manual low-poly reference:
#   build -> compare with the manual model -> check -> verify -> review sheets.
#
#   dev/skelretopo_round.sh SRC.fbx OUTDIR [SPEC] [MANUAL.json]
#
#   SPEC     triangle budgets, default 'character:1400+760,axe:530,crossbow:448' (范海辛, round 9d)
#   MANUAL   manual low-poly description for dev/skelretopo_cmp.py (skip the comparison if empty)
#   PY       python with bpy (default: python3)
#
# Output in OUTDIR: FBX + BC/N/AO, report.json, previews/, log.txt, cmp/ (+ cmp/j/*.jpg),
# check.log, verify.log, cmp/key_*.jpg and cmp/key9_*.jpg (columns: manual | ours | high).
# A round passes only if every CHECK line says OK and every VERIFY line says PASS;
# then look at the key sheets yourself (docs/skelretopo_methods.md, "看图清单").
set -e
ROOT=${0:A:h:h}
PY=${PY:-python3}
SRC=$1; OUT=$2; SPEC=${3:-character:1400+760,axe:530,crossbow:448}; MANUAL=${4:-}
[[ -z $SRC || -z $OUT ]] && { sed -n '2,13p' $0; exit 1; }
mkdir -p $OUT
NAME=${SRC:t:r}

$PY $ROOT/dev/skelretopo_vh.py $SRC $OUT "$SPEC" > $OUT/log.txt 2>&1 || { tail -30 $OUT/log.txt; exit 1; }
if [[ -n $MANUAL ]]; then
  $PY $ROOT/dev/skelretopo_cmp.py $OUT $MANUAL $OUT/cmp > $OUT/cmp.log 2>&1
  sed -n '/^### /,$p' $OUT/cmp.log
  mkdir -p $OUT/cmp/j
  for f in $OUT/cmp/*.png(N); do sips -s format jpeg -Z 1200 $f --out $OUT/cmp/j/${f:t:r}.jpg >/dev/null 2>&1 || true; done
fi
$PY $ROOT/dev/skelretopo_check.py $OUT > $OUT/check.log 2>&1 || true
grep "^CHECK" $OUT/check.log | cut -c1-200
$PY $ROOT/dev/retopo_verify.py $OUT $SRC > $OUT/verify.log 2>&1 || true
grep "^VERIFY" $OUT/verify.log | cut -c1-120

[[ -z $MANUAL ]] && exit 0
C=$OUT/cmp; P=$OUT/previews/${NAME}_character
M() { $PY $ROOT/dev/montage.py "$@" 2>&1 | grep -i "error\|missing" || true; }
M $C/key_hand_bracer.jpg $C/cmp_character_hand_{wire,tex,bottom_wire,top_wire}.png $C/cmp_character_bracer_{wire,tex,top_wire,back_wire}.png
M $C/key_weapons.jpg $C/cmp_axe_{front_wire,front_tex,zoom_upper_wire,left_wire}.png $C/cmp_crossbow_{front_wire,front_tex,zoom_upper_wire,left_wire}.png
M $C/key_char_closeups.jpg $C/cmp_character_hat_{wire,34_wire,tex}.png $C/cmp_character_cape_{back,side}_wire.png $C/cmp_character_quiver_{right_wire,34_wire,tex}.png
M $C/key_pose.jpg ${P}_pose_fingers_{front,left}.png ${P}_pose_elbow90_front.png ${P}_pose_knee90_left.png ${P}_pose_shoulder_up_front.png
M $C/key9_fingers.jpg $C/cmp_character_fingers_{top_wire,side_wire}.png $C/cmp_character_hand_{wire,bottom_wire}.png $C/cmp_character_fingers_top_tex.png
M $C/key9_hair.jpg $C/cmp_character_neck_{back,side,34}_wire.png $C/cmp_character_neck_{back,side,34}_tex.png
M $C/key9_crossbow.jpg $C/cmp_crossbow_{zoom_end_wire,zoom_end_tex,front_wire,front_tex,zoom_upper_wire}.png
M $C/key9_boots.jpg $C/cmp_character_boot_{side,front}_wire.png $C/cmp_character_boots_{top,bottom}_wire.png $C/cmp_character_boot_side_tex.png $C/cmp_character_boots_tex.png
ls $C/key*.jpg
