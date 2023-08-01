declare -A tasks
# multilingual
tasks["xcopa"]=xcopa_*
tasks["xstory_cloze"]=xstory_cloze_*
tasks["xwinograd"]=xwinograd_*
tasks["pawsx"]=pawsx_*
tasks["xnli"]=xnli_*

# mt nllb
tasks["xstory_cloze-mt"]=xstory_cloze-mt_nllb*
tasks["xwinograd-mt"]=xwinograd-mt_nllb*
tasks["xcopa-mt"]=xcopa-mt_nllb*
tasks["pawsx-mt"]=pawsx-mt_nllb*
# tasks["xnli-mt"]=xnli-mt_nllb*
tasks["xnli-mt"]=xnli-mt_nllb-200-3.3B*
tasks["xnli-mt-all"]=xnli-mt_nllb-200-1.3B*,xnli-mt_nllb-200-distilled*

# mt models
tasks["xstory_cloze-mt_bloom"]=xstory_cloze-mt_bloom-560m*,xstory_cloze-mt_bloom-1b7*,xstory_cloze-mt_bloom-3b*,xstory_cloze-mt_bloom-7b1*
tasks["xstory_cloze-mt_xglm"]=xstory_cloze-mt_xglm-564M*,xstory_cloze-mt_xglm-1.7B*,xstory_cloze-mt_xglm-2.9B*,xstory_cloze-mt_xglm-7.5B*
tasks["xcopa-mt_bloom"]=xcopa-mt_bloom-560m*,xcopa-mt_bloom-1b7*,xcopa-mt_bloom-3b*,xcopa-mt_bloom-7b1*
tasks["xcopa-mt_xglm"]=xcopa-mt_xglm-564M*,xcopa-mt_xglm-1.7B*,xcopa-mt_xglm-2.9B*,xcopa-mt_xglm-7.5B*
tasks["pawsx-mt_bloom"]=pawsx-mt_bloom-560m*,pawsx-mt_bloom-1b7*,pawsx-mt_bloom-3b*,pawsx-mt_bloom-7b1*
tasks["pawsx-mt_xglm"]=pawsx-mt_xglm-564M*,pawsx-mt_xglm-1.7B*,pawsx-mt_xglm-2.9B*,pawsx-mt_xglm-7.5B*
tasks["xnli-mt_bloom"]=xnli-mt_bloom-560m*,xnli-mt_bloom-1b7*,xnli-mt_bloom-3b*,xnli-mt_bloom-7b1*
tasks["xnli-mt_xglm"]=xnli-mt_xglm-564M*,xnli-mt_xglm-1.7B*,xnli-mt_xglm-2.9B*,xnli-mt_xglm-7.5B*