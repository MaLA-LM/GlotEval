# metrics/translation_metrics.py
import sacrebleu
import json
import pandas as pd
from sacrebleu.metrics import CHRF

from sacrebleu.metrics import BLEU, CHRF

def compute_bleu_score(predictions, references):
    """
    Computes the BLEU score for the predictions against the references using the Flores-200 tokenizer.

    Args:
        predictions (list): List of predicted translations.
        references (list): List of reference translations.

    Returns:
        bleu_score (float): The BLEU score.
    """
    bleu = BLEU(tokenize="flores200")
    bleu_score = bleu.corpus_score(predictions, [references])
    return bleu_score.score

def compute_chrf_score(predictions, references):
    """
    Computes the CHRF++ score for the predictions against the references using word_order=2.

    Args:
        predictions (list): List of predicted translations.
        references (list): List of reference translations.

    Returns:
        chrf_score (float): The CHRF++ score.
    """
    chrf = CHRF(word_order=2)
    chrf_score = chrf.corpus_score(predictions, [references])
    return chrf_score.score

def compute_chrf_segments_score(predictions, references):
    """
    Computes CHRF scores at the segment level.

    Args:
        predictions (list): List of predicted translations.
        references (list): List of reference translations.

    Returns:
        tuple: (Average CHRF score, List of individual segment CHRF scores)
    """
    scorer = CHRF()
    segment_scores = []
    
    for hyp, ref in zip(predictions, references):
            if not isinstance(ref, list):
                ref = [ref]
            score = scorer.sentence_score(hyp, ref)
            segment_score = json.loads(score.format(signature=str(scorer.get_signature()), is_json=True))["score"]
            segment_scores.append(segment_score)
        
    
    return sum(segment_scores) / len(segment_scores) if segment_scores else 0, segment_scores

def compute_mmhb(df):
    df = df.fillna('')

    df_masc = df[df['masculine'] != '']
    df_fem  = df[df['feminine'] != '']
    df_both = df[df['both']     != '']

    # Extract predicted translations
    masculine_translations = df_masc['translation'].tolist()
    feminine_translations  = df_fem['translation'].tolist()
    both_translations      = df_both['translation'].tolist()

    # Convert stringified references into Python lists 
    masculine_refs = [eval(ref_str) for ref_str in df_masc['masculine']]
    feminine_refs  = [eval(ref_str) for ref_str in df_fem['feminine']]
    both_refs      = [eval(ref_str) for ref_str in df_both['both']]

    chrf_masculine, chrf_masculine_segments = compute_chrf_segments_score(masculine_translations, masculine_refs)
    chrf_feminine, chrf_feminine_segments   = compute_chrf_segments_score(feminine_translations,  feminine_refs)
    chrf_both, chrf_both_segments           = compute_chrf_segments_score(both_translations,      both_refs)

    return  {
        # Overall CHRF scores
        'chrfs_masculine': chrf_masculine,
        'chrfs_feminine':  chrf_feminine,
        'chrfs_both':      chrf_both,
        # Segment-level CHRF
        # 'chrfs_masculine_segments': chrf_masculine_segments,
        # 'chrfs_feminine_segments':  chrf_feminine_segments,
        # 'chrfs_both_segments':      chrf_both_segments ,

    }
def compute_comet_score(sources, predictions, references, src_lang, tgt_lang, model):
    """
    Computes the COMET score for the predictions against the references, also using the source sentences.

    Args:
        sources (list): List of source translations.
        predictions (list): List of predicted translations.
        references (list): List of reference translations.

    Returns:
        comet_score (float): The COMET score.
    """
    from comet import download_model, load_from_checkpoint
    
    if src_lang and tgt_lang in comet_languages:
        print("Both source and target languages are included in COMET. Let's proceed.")
    else:
        print("Not all languages are included in COMET. Aborting this metric computation.")
        return "--"

    model_path = download_model(model) # The model should be optional
    model = load_from_checkpoint(model_path)

    data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(sources, predictions, references)]
    comet = model.predict(data, batch_size=8, gpus=1)

    return comet.system_score


# Languages covered by comet
comet_languages = [
    "afr_Latn",  # Afrikaans
    "sqi_Latn",  # Albanian
    "amh_Ethi",  # Amharic
    "ara_Arab",  # Arabic
    "hye_Armn",  # Armenian
    "asm_Beng",  # Assamese
    "azj_Latn",  # Azerbaijani
    "eus_Latn",  # Basque
    "bel_Cyrl",  # Belarusian
    "ben_Beng",  # Bengali
    "ben_Latn",  # Bengali Romanized
    "bos_Latn",  # Bosnian
    "bre_Latn",  # Breton
    "bul_Cyrl",  # Bulgarian
    "mya_Mymr",  # Burmese
    "cat_Latn",  # Catalan
    "zho_Hans",  # Chinese (Simplified)
    "zho_Hant",  # Chinese (Traditional)
    "hrv_Latn",  # Croatian
    "ces_Latn",  # Czech
    "dan_Latn",  # Danish
    "nld_Latn",  # Dutch
    "eng_Latn",  # English
    "epo_Latn",  # Esperanto
    "est_Latn",  # Estonian
    "fil_Latn",  # Filipino
    "fin_Latn",  # Finnish
    "fra_Latn",  # French
    "glg_Latn",  # Galician
    "kat_Geo",   # Georgian
    "deu_Latn",  # German
    "ell_Grek",  # Greek
    "guj_Gujr",  # Gujarati
    "hau_Latn",  # Hausa
    "heb_Hebr",  # Hebrew
    "hin_Deva",  # Hindi
    "hin_Latn",  # Hindi Romanized
    "hun_Latn",  # Hungarian
    "isl_Latn",  # Icelandic
    "ind_Latn",  # Indonesian
    "gle_Latn",  # Irish
    "ita_Latn",  # Italian
    "jpn_Jpan",  # Japanese
    "jav_Latn",  # Javanese
    "kan_Knda",  # Kannada
    "kaz_Cyrl",  # Kazakh
    "khm_Khmr",  # Khmer
    "kor_Hang",  # Korean
    "kmr_Latn",  # Kurdish (Kurmanji)
    "kir_Cyrl",  # Kyrgyz
    "lao_Laoo",  # Lao
    "lat_Latn",  # Latin
    "lav_Latn",  # Latvian
    "lit_Latn",  # Lithuanian
    "mkd_Cyrl",  # Macedonian
    "mlg_Latn",  # Malagasy
    "zsm_Latn",  # Malay
    "mal_Mlym",  # Malayalam
    "mar_Deva",  # Marathi
    "mon_Cyrl",  # Mongolian
    "nep_Deva",  # Nepali
    "nob_Latn",  # Norwegian
    "ori_Orya",  # Oriya
    "orm_Latn",  # Oromo
    "pus_Arab",  # Pashto
    "fas_Arab",  # Persian
    "pol_Latn",  # Polish
    "por_Latn",  # Portuguese
    "pan_Guru",  # Punjabi
    "ron_Latn",  # Romanian
    "rus_Cyrl",  # Russian
    "san_Deva",  # Sanskrit
    "gla_Latn",  # Scottish Gaelic
    "srp_Cyrl",  # Serbian
    "snd_Arab",  # Sindhi
    "sin_Sinh",  # Sinhala
    "slk_Latn",  # Slovak
    "slv_Latn",  # Slovenian
    "som_Latn",  # Somali
    "spa_Latn",  # Spanish
    "sun_Latn",  # Sundanese
    "swa_Latn",  # Swahili
    "swe_Latn",  # Swedish
    "tam_Taml",  # Tamil
    "tam_Latn",  # Tamil Romanized
    "tel_Telu",  # Telugu
    "tel_Latn",  # Telugu Romanized
    "tha_Thai",  # Thai
    "tur_Latn",  # Turkish
    "ukr_Cyrl",  # Ukrainian
    "urd_Arab",  # Urdu
    "urd_Latn",  # Urdu Romanized
    "uig_Arab",  # Uyghur
    "uzn_Latn",  # Uzbek
    "vie_Latn",  # Vietnamese
    "cym_Latn",  # Welsh
    "fry_Latn",  # Western Frisian
    "xho_Latn",  # Xhosa
    "yid_Hebr"   # Yiddish
]