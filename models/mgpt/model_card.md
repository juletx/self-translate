---
license: apache-2.0
language:
- en
- az
- sw
- af
- ar
- ba
- be
- bxr
- bg
- bn
- cv
- hy
- da
- de
- el
- es
- eu
- fa
- fi
- fr
- he
- hi
- hu
- kk
- id
- it
- ja
- ka
- ky
- ko
- lt
- lv
- mn
- ml
- os
- mr
- ms
- my
- nl
- ro
- pl
- pt
- sah
- ru
- tg
- sv
- ta
- te
- tk
- th
- tr
- tl
- tt
- tyv
- uk
- en
- ur
- vi
- uz
- yo
- zh
- xal
pipeline_tag: text-generation
tags:
- multilingual
- PyTorch
- Transformers
- gpt3
- gpt2
- Deepspeed
- Megatron
datasets:
- mc4
- wikipedia
thumbnail: "https://github.com/sberbank-ai/mgpt"
---

# Multilingual GPT model

We introduce a family of autoregressive GPT-like models with 1.3 billion parameters trained on 60 languages from 25 language families using Wikipedia and Colossal Clean Crawled Corpus. 

We reproduce the GPT-3 architecture using GPT-2 sources and the sparse attention mechanism, [Deepspeed](https://github.com/microsoft/DeepSpeed) and [Megatron](https://github.com/NVIDIA/Megatron-LM) frameworks allows us to effectively parallelize the training and inference steps. The resulting models show performance on par with the recently released [XGLM](https://arxiv.org/pdf/2112.10668.pdf) models at the same time covering more languages and enhancing NLP possibilities for low resource languages. 

## Code
The source code for the mGPT XL model is available on [Github](https://github.com/sberbank-ai/mgpt)

## Paper
 mGPT: Few-Shot Learners Go Multilingual
 
 [Abstract](https://arxiv.org/abs/2204.07580) [PDF](https://arxiv.org/pdf/2204.07580.pdf)

 ![](https://habrastorage.org/webt/1q/ru/yt/1qruytul6m2m-upyk9frq3pgrds.png)

 ```
@misc{https://doi.org/10.48550/arxiv.2204.07580,
  doi = {10.48550/ARXIV.2204.07580},
  
  url = {https://arxiv.org/abs/2204.07580},
  
  author = {Shliazhko, Oleh and Fenogenova, Alena and Tikhonova, Maria and Mikhailov, Vladislav and Kozlova, Anastasia and Shavrina, Tatiana},
  
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences, I.2; I.2.7, 68-06, 68-04, 68T50, 68T01},
  
  title = {mGPT: Few-Shot Learners Go Multilingual},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}

 ```


## Languages

Model supports 60 languages: 

ISO codes:
```az, sw, af, ar, ba, be, bxr, bg, bn, cv, hy, da, de, el, es, eu, fa, fi, fr, he, hi, hu, kk, id, it, ja, ka, ky, ko, lt, lv, mn, ml, os, mr, ms, my, nl, ro, pl, pt, sah, ru, tg, sv, ta, te, tk, th, tr, tl, tt, tyv, uk, en, ur, vi, uz, yo, zh, xal```


Languages:

```Afrikaans, Azerbaijani, Belarusian, Bengali, Chuvash, German, English, Basque, Finnish, Hebrew (modern), Hungarian, Indonesian, Japanese, Kazakh, Kirghiz, Kyrgyz, Latvian, Mongolian, Malay, Dutch, Polish, Romanian, Moldavan, Yakut, Swahili, Telugu, Thai, Turkish, Tuvinian, Urdu, Vietnamese, Yoruba, Arabic, Bashkir, Bulgarian, Buriat, Danish, Greek, Modern, Spanish; Castilian, Persian, French, Hindi, Armenian, Italian, Georgian, Korean, Lithuanian, Malayalam, Marathi, Burmese, Ossetian, Ossetic, Portuguese, Russian, Swedish, Tamil, Tajik, Turkmen, Tatar, Ukrainian, Uzbek, Kalmyk, Chinese```

## Training Data Statistics

 - Size: 488 Billion UTF characters


<img style="text-align:center; display:block;" src="https://huggingface.co/sberbank-ai/mGPT/resolve/main/stats.png">
"General training corpus statistics"


## Details
The model was trained with sequence length 512 using Megatron and Deepspeed libs by [SberDevices](https://sberdevices.ru/) team on a dataset of 600 GB of texts in 60 languages. The model has seen 440 billion BPE tokens in total.

Total training time was around 12 days on 256 Nvidia V100 GPUs.  
