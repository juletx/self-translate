declare -A tasks
tasks["glue"]=cola,mnli,mnli_mismatched,mrpc,rte,qnli,qqp,sst,wnli # stsb not implemented yet

tasks["superglue"]=boolq,cb,copa,multirc,record,wic,wsc

tasks["language_modeling"]=wikitext,lambada_*,pile_*
tasks["lambada"]=lambada_*
tasks["lambada_openai"]=lambada_openai,lambada_openai_cloze
tasks["lambada_standard"]=lambada_standard,lambada_standard_cloze
tasks["lambada_openai_mt"]=lambada_openai_mt_*
tasks["pile"]=pile_*

tasks["common_sense_reasoning"]=boolq,piqa,winogrande,hellaswag,arc_easy,arc_challenge,openbookqa,copa,prost,mc_taco,swag # story_cloze_2016
tasks["xcopa"]=xcopa_*
tasks["xstory_cloze"]=xstory_cloze_*
tasks["xwinograd"]=xwinograd_*

tasks["reading_comprehension"]=race,coqa,drop # race-middle, quac

tasks["anli"]=anli_*

tasks["unscramble"]=anagrams1,anagrams2,cycle_letters,random_insertion,reversed_words

tasks["blimp"]=blimp_*

tasks["machine_translation"]=wmt*,iwslt*
tasks["machine_translation_gpt3"]=wmt14*,wmt16*
tasks["machine_translation_harness"]=wmt20*,iwslt17*

tasks["science"]=pubmedqa,sciq,qasper,qa4mre_*

tasks["dialogue"]=mutual*

tasks["question_answering"]=triviaqa,squad2,truthfulqa_mc,webqs,headqa_*,logiqa # naturalqs
tasks["xquad"]=xquad_*
tasks["mlqa"]=mlqa.*
tasks["tydiqa"]=tydiqa_*

tasks["arithmetic"]=arithmetic_*
tasks["mathematical_reasoning"]=gsm8k,mathqa,drop,math_*
tasks["mathematical_reasoning_mgsm"]=mgsm_*

tasks["human_alignment"]=toxigen,truthfulqa_gen,ethics_*,crows_pairs_english_*,crows_pairs_french_* # winogender
tasks["ethics"]=ethics_*
tasks["crows_pairs_english"]=crows_pairs_english_*
tasks["crows_pairs_french"]=crows_pairs_french_*

tasks["mmlu"]=hendrycksTest-*