"""
Utility module for language codes and mappings between different formats.
"""

# Mapping from Microsoft language codes to ISO 639-3_Script format
MS_TO_ISO = {
    'af': 'afr_Latn',     # Afrikaans
    'am': 'amh_Ethi',     # Amharic
    'ar': 'ara_Arab',     # Arabic
    'as': 'asm_Beng',     # Assamese
    'az': 'aze_Latn',     # Azerbaijani
    'ba': 'bak_Cyrl',     # Bashkir
    'bn': 'ben_Beng',     # Bengali
    'bho': 'bho_Deva',    # Bhojpuri
    'bo': 'bod_Tibt',     # Tibetan
    'brx': 'brx_Deva',    # Bodo
    'bs': 'bos_Latn',     # Bosnian
    'bg': 'bul_Cyrl',     # Bulgarian
    'ca': 'cat_Latn',     # Catalan
    'cs': 'ces_Latn',     # Czech
    'cy': 'cym_Latn',     # Welsh
    'da': 'dan_Latn',     # Danish
    'de': 'deu_Latn',     # German
    'dv': 'div_Thaa',     # Divehi
    'doi': 'doi_Deva',    # Dogri
    'dsb': 'dsb_Latn',    # Lower Sorbian
    'el': 'ell_Grek',     # Greek
    'en': 'eng_Latn',     # English
    'es': 'spa_Latn',     # Spanish
    'et': 'est_Latn',     # Estonian
    'eu': 'eus_Latn',     # Basque
    'fa': 'fas_Arab',     # Persian
    'fi': 'fin_Latn',     # Finnish
    'fj': 'fij_Latn',     # Fijian
    'fil': 'fil_Latn',    # Filipino
    'fo': 'fao_Latn',     # Faroese
    'fr': 'fra_Latn',     # French (France)
    'fr-CA': 'fra_Latn_CA',  # French (Canada)
    'gl': 'glg_Latn',     # Galician
    'ga': 'gle_Latn',     # Irish
    'gom': 'gom_Deva',    # Konkani (Goan)
    'gu': 'guj_Gujr',     # Gujarati
    'ha': 'hau_Latn',     # Hausa
    'he': 'heb_Hebr',     # Hebrew
    'hi': 'hin_Deva',     # Hindi
    'hne': 'hne_Deva',    # Chhattisgarhi
    'hr': 'hrv_Latn',     # Croatian
    'hsb': 'hsb_Latn',    # Upper Sorbian
    'ht': 'hat_Latn',     # Haitian Creole
    'hu': 'hun_Latn',     # Hungarian
    'hy': 'hye_Armn',     # Armenian
    'id': 'ind_Latn',     # Indonesian
    'ig': 'ibo_Latn',     # Igbo
    'ikt': 'ikt_Latn',    # Inuinnaqtun
    'iu-Cans': 'iku_Cans', # Inuktitut (Canadian Aboriginal) - may cause issues
    'iu-Latn': 'iku_Latn', # Inuktitut (Latin)
    'is': 'isl_Latn',     # Icelandic
    'it': 'ita_Latn',     # Italian
    'ja': 'jpn_Jpan',     # Japanese
    'ka': 'kat_Geor',     # Georgian
    'kk': 'kaz_Cyrl',     # Kazakh
    'km': 'khm_Khmr',     # Khmer
    'kn': 'kan_Knda',     # Kannada
    'ko': 'kor_Hang',     # Korean
    'ks': 'kas_Arab',     # Kashmiri
    'ku': 'kur_Arab',     # Kurdish (Arabic)
    'kmr': 'kmr_Latn',    # Kurdish (Northern)
    'ky': 'kir_Cyrl',     # Kyrgyz
    'ln': 'lin_Latn',     # Lingala
    'lo': 'lao_Laoo',     # Lao
    'lt': 'lit_Latn',     # Lithuanian
    'lug': 'lug_Latn',    # Luganda
    'lv': 'lav_Latn',     # Latvian
    'lzh': 'lzh_Hans',    # Literary Chinese
    'mai': 'mai_Deva',    # Maithili
    'mg': 'mlg_Latn',     # Malagasy
    'mi': 'mri_Latn',     # Maori
    'mk': 'mkd_Cyrl',     # Macedonian
    'ml': 'mal_Mlym',     # Malayalam
    'mn-Cyrl': 'mon_Cyrl', # Mongolian (Cyrillic)
    'mn-Mong': 'mon_Mong', # may cause issues
    'mni': 'mni_Beng',    # Manipuri
    'mr': 'mar_Deva',     # Marathi
    'ms': 'msa_Latn',     # Malay
    'mt': 'mlt_Latn',     # Maltese
    'mww': 'mww_Latn',    # Hmong Daw
    'my': 'mya_Mymr',     # Myanmar
    'nb': 'nob_Latn',     # Norwegian Bokmål
    'ne': 'nep_Deva',     # Nepali
    'nl': 'nld_Latn',     # Dutch
    'nso': 'nso_Latn',    # Northern Sotho
    'nya': 'nya_Latn',    # Chichewa
    'or': 'ori_Orya',     # Odia
    'otq': 'otq_Latn',    # Querétaro Otomi
    'pa': 'pan_Guru',     # Punjabi
    'pl': 'pol_Latn',     # Polish
    'prs': 'prs_Arab',    # Dari
    'ps': 'pus_Arab',     # Pashto
    'pt': 'por_Latn',     # Portuguese (Brazil)
    'pt-PT': 'por_Latn_PT', # Portuguese (Portugal)
    'ro': 'ron_Latn',     # Romanian
    'ru': 'rus_Cyrl',     # Russian
    'run': 'run_Latn',    # Kirundi
    'rw': 'kin_Latn',     # Kinyarwanda
    'sd': 'snd_Arab',     # Sindhi
    'si': 'sin_Sinh',     # Sinhala
    'sk': 'slk_Latn',     # Slovak
    'sl': 'slv_Latn',     # Slovenian
    'sm': 'smo_Latn',     # Samoan
    'sn': 'sna_Latn',     # Shona
    'so': 'som_Latn',     # Somali
    'sq': 'sqi_Latn',     # Albanian
    'sr-Cyrl': 'srp_Cyrl', # Serbian (Cyrillic)
    'sr-Latn': 'srp_Latn', # Serbian (Latin)
    'st': 'sot_Latn',     # Sesotho
    'sv': 'swe_Latn',     # Swedish
    'sw': 'swa_Latn',     # Swahili
    'ta': 'tam_Taml',     # Tamil
    'te': 'tel_Telu',     # Telugu
    'th': 'tha_Thai',     # Thai
    'ti': 'tir_Ethi',     # Tigrinya
    'tk': 'tuk_Latn',     # Turkmen
    'tlh-Latn': 'tlh_Latn', # Klingon (Latin) - may cause issues
    'tlh-Piqd': 'tlh_Piqd', # Klingon (pIqaD) - may cause issues
    'tn': 'tsn_Latn',     # Tswana
    'to': 'ton_Latn',     # Tongan
    'tr': 'tur_Latn',     # Turkish
    'tt': 'tat_Cyrl',     # Tatar
    'ty': 'tah_Latn',     # Tahitian
    'ug': 'uig_Arab',     # Uyghur
    'uk': 'ukr_Cyrl',     # Ukrainian
    'ur': 'urd_Arab',     # Urdu
    'uz': 'uzb_Latn',     # Uzbek
    'vi': 'vie_Latn',     # Vietnamese
    'xh': 'xho_Latn',     # Xhosa
    'yo': 'yor_Latn',     # Yoruba
    'yua': 'yua_Latn',    # Yucatec Maya
    'yue': 'yue_Hant',    # Cantonese
    'zh-Hans': 'zho_Hans', # Simplified Chinese
    'zh-Hant': 'zho_Hant', # Traditional Chinese
    'zu': 'zul_Latn',     # Zulu
}

# Create the reverse mapping: ISO format to Microsoft format
ISO_TO_MS = {v: k for k, v in MS_TO_ISO.items()}

# Basic language names mapping
LANGUAGE_NAMES = {
    'afr_Latn': 'Afrikaans',
    'amh_Ethi': 'Amharic',
    'ara_Arab': 'Arabic',
    'asm_Beng': 'Assamese',
    'aze_Latn': 'Azerbaijani',
    'bak_Cyrl': 'Bashkir',
    'ben_Beng': 'Bengali',
    'bho_Deva': 'Bhojpuri',
    'bod_Tibt': 'Tibetan',
    'brx_Deva': 'Bodo',
    'bos_Latn': 'Bosnian',
    'bul_Cyrl': 'Bulgarian',
    'cat_Latn': 'Catalan',
    'ces_Latn': 'Czech',
    'cym_Latn': 'Welsh',
    'dan_Latn': 'Danish',
    'deu_Latn': 'German',
    'div_Thaa': 'Divehi',
    'doi_Deva': 'Dogri',
    'dsb_Latn': 'Lower Sorbian',
    'ell_Grek': 'Greek',
    'eng_Latn': 'English',
    'spa_Latn': 'Spanish',
    'est_Latn': 'Estonian',
    'eus_Latn': 'Basque',
    'fas_Arab': 'Persian',
    'fin_Latn': 'Finnish',
    'fij_Latn': 'Fijian',
    'fil_Latn': 'Filipino',
    'fao_Latn': 'Faroese',
    'fra_Latn': 'French',
    'fra_Latn_CA': 'French (Canada)',
    'glg_Latn': 'Galician',
    'gle_Latn': 'Irish',
    'gom_Deva': 'Konkani',
    'guj_Gujr': 'Gujarati',
    'hau_Latn': 'Hausa',
    'heb_Hebr': 'Hebrew',
    'hin_Deva': 'Hindi',
    'hne_Deva': 'Chhattisgarhi',
    'hrv_Latn': 'Croatian',
    'hsb_Latn': 'Upper Sorbian',
    'hat_Latn': 'Haitian Creole',
    'hun_Latn': 'Hungarian',
    'hye_Armn': 'Armenian',
    'ibo_Latn': 'Igbo',
    'ind_Latn': 'Indonesian',
    'iku_Cans': 'Inuktitut (Canadian Aboriginal)',
    'ikt_Latn': 'Inuinnaqtun',
    'iku_Latn': 'Inuktitut',
    'isl_Latn': 'Icelandic',
    'ita_Latn': 'Italian',
    'jpn_Jpan': 'Japanese',
    'kat_Geor': 'Georgian',
    'kaz_Cyrl': 'Kazakh',
    'khm_Khmr': 'Khmer',
    'kan_Knda': 'Kannada',
    'kor_Hang': 'Korean',
    'kas_Arab': 'Kashmiri',
    'kur_Arab': 'Kurdish',
    'kmr_Latn': 'Kurdish (Northern)',
    'kir_Cyrl': 'Kyrgyz',
    'lin_Latn': 'Lingala',
    'lao_Laoo': 'Lao',
    'lit_Latn': 'Lithuanian',
    'lug_Latn': 'Luganda',
    'lav_Latn': 'Latvian',
    'lzh_Hans': 'Literary Chinese',
    'mai_Deva': 'Maithili',
    'mlg_Latn': 'Malagasy',
    'mri_Latn': 'Maori',
    'mkd_Cyrl': 'Macedonian',
    'mal_Mlym': 'Malayalam',
    'mon_Cyrl': 'Mongolian',
    'mon_Mong': 'Mongolian (Traditional Mongolian)',
    'mni_Beng': 'Manipuri',
    'mar_Deva': 'Marathi',
    'msa_Latn': 'Malay',
    'mlt_Latn': 'Maltese',
    'mww_Latn': 'Hmong Daw',
    'mya_Mymr': 'Myanmar',
    'nob_Latn': 'Norwegian',
    'nep_Deva': 'Nepali',
    'nld_Latn': 'Dutch',
    'nso_Latn': 'Northern Sotho',
    'nya_Latn': 'Chichewa',
    'ori_Orya': 'Odia',
    'otq_Latn': 'Querétaro Otomi',
    'pan_Guru': 'Punjabi',
    'pol_Latn': 'Polish',
    'prs_Arab': 'Dari',
    'pus_Arab': 'Pashto',
    'por_Latn': 'Portuguese (Brazil)',
    'por_Latn_PT': 'Portuguese (Portugal)',
    'ron_Latn': 'Romanian',
    'rus_Cyrl': 'Russian',
    'run_Latn': 'Kirundi',
    'kin_Latn': 'Kinyarwanda',
    'snd_Arab': 'Sindhi',
    'sin_Sinh': 'Sinhala',
    'slk_Latn': 'Slovak',
    'slv_Latn': 'Slovenian',
    'smo_Latn': 'Samoan',
    'sna_Latn': 'Shona',
    'som_Latn': 'Somali',
    'sqi_Latn': 'Albanian',
    'srp_Cyrl': 'Serbian (Cyrillic)',
    'srp_Latn': 'Serbian (Latin)',
    'sot_Latn': 'Sesotho',
    'swe_Latn': 'Swedish',
    'swa_Latn': 'Swahili',
    'tam_Taml': 'Tamil',
    'tel_Telu': 'Telugu',
    'tha_Thai': 'Thai',
    'tir_Ethi': 'Tigrinya',
    'tlh_Latn': 'Klingon (Latin)',
    'tlh_Piqd': 'Klingon (pIqaD)',
    'tuk_Latn': 'Turkmen',
    'tsn_Latn': 'Tswana',
    'ton_Latn': 'Tongan',
    'tur_Latn': 'Turkish',
    'tat_Cyrl': 'Tatar',
    'tah_Latn': 'Tahitian',
    'uig_Arab': 'Uyghur',
    'ukr_Cyrl': 'Ukrainian',
    'urd_Arab': 'Urdu',
    'uzb_Latn': 'Uzbek',
    'vie_Latn': 'Vietnamese',
    'xho_Latn': 'Xhosa',
    'yor_Latn': 'Yoruba',
    'yua_Latn': 'Yucatec Maya',
    'yue_Hant': 'Cantonese',
    'zho_Hans': 'Chinese (Simplified)',
    'zho_Hant': 'Chinese (Traditional)',
    'zul_Latn': 'Zulu',
}

# More detailed language names with script information
LANGUAGE_NAMES_WITH_SCRIPT = {
    'afr_Latn': 'Afrikaans (written in Latin script)',
    'amh_Ethi': 'Amharic (written in the Ethiopic script)',
    'ara_Arab': 'Arabic (written in the Arabic script)',
    'asm_Beng': 'Assamese (written in the Bengali script)',
    'aze_Latn': 'Azerbaijani (written in Latin script)',
    'bak_Cyrl': 'Bashkir (written in the Cyrillic script)',
    'ben_Beng': 'Bengali (written in the Bengali script)',
    'bho_Deva': 'Bhojpuri (written in the Devanagari script)',
    'bod_Tibt': 'Tibetan (written in the Tibetan script)',
    'brx_Deva': 'Bodo (written in the Devanagari script)',
    'bos_Latn': 'Bosnian (written in Latin script)',
    'bul_Cyrl': 'Bulgarian (written in the Cyrillic script)',
    'cat_Latn': 'Catalan (written in Latin script)',
    'ces_Latn': 'Czech (written in Latin script)',
    'cym_Latn': 'Welsh (written in Latin script)',
    'dan_Latn': 'Danish (written in Latin script)',
    'deu_Latn': 'German (written in Latin script)',
    'div_Thaa': 'Divehi (written in the Thaana script)',
    'doi_Deva': 'Dogri (written in the Devanagari script)',
    'dsb_Latn': 'Lower Sorbian (written in Latin script)',
    'ell_Grek': 'Greek (written in the Greek script)',
    'eng_Latn': 'English (written in Latin script)',
    'spa_Latn': 'Spanish (written in Latin script)',
    'est_Latn': 'Estonian (written in Latin script)',
    'eus_Latn': 'Basque (written in Latin script)',
    'fas_Arab': 'Persian (written in the Arabic script)',
    'fin_Latn': 'Finnish (written in Latin script)',
    'fij_Latn': 'Fijian (written in Latin script)',
    'fil_Latn': 'Filipino (written in Latin script)',
    'fao_Latn': 'Faroese (written in Latin script)',
    'fra_Latn': 'French (written in Latin script)',
    'fra_Latn_CA': 'French (Canada) (written in Latin script)',
    'glg_Latn': 'Galician (written in Latin script)',
    'gle_Latn': 'Irish (written in Latin script)',
    'gom_Deva': 'Konkani (written in the Devanagari script)',
    'guj_Gujr': 'Gujarati (written in the Gujarati script)',
    'hau_Latn': 'Hausa (written in Latin script)',
    'heb_Hebr': 'Hebrew (written in the Hebrew script)',
    'hin_Deva': 'Hindi (written in the Devanagari script)',
    'hne_Deva': 'Chhattisgarhi (written in the Devanagari script)',
    'hrv_Latn': 'Croatian (written in Latin script)',
    'hsb_Latn': 'Upper Sorbian (written in Latin script)',
    'hat_Latn': 'Haitian Creole (written in Latin script)',
    'hun_Latn': 'Hungarian (written in Latin script)',
    'hye_Armn': 'Armenian (written in the Armenian script)',
    'ibo_Latn': 'Igbo (written in Latin script)',
    'ind_Latn': 'Indonesian (written in Latin script)',
    'iku_Cans': 'Inuktitut (written in Canadian Aboriginal Syllabics)',
    'ikt_Latn': 'Inuinnaqtun (written in Latin script)',
    'iku_Latn': 'Inuktitut (written in Latin script)',
    'isl_Latn': 'Icelandic (written in Latin script)',
    'ita_Latn': 'Italian (written in Latin script)',
    'jpn_Jpan': 'Japanese (written in Japanese script (Kanji and Kana))',
    'kat_Geor': 'Georgian (written in the Georgian script)',
    'kaz_Cyrl': 'Kazakh (written in the Cyrillic script)',
    'khm_Khmr': 'Khmer (written in the Khmer script)',
    'kan_Knda': 'Kannada (written in the Kannada script)',
    'kor_Hang': 'Korean (written in Hangul)',
    'kas_Arab': 'Kashmiri (written in the Arabic script)',
    'kur_Arab': 'Kurdish (written in the Arabic script)',
    'kmr_Latn': 'Kurdish (Northern) (written in Latin script)',
    'kir_Cyrl': 'Kyrgyz (written in the Cyrillic script)',
    'lin_Latn': 'Lingala (written in Latin script)',
    'lao_Laoo': 'Lao (written in the Lao script)',
    'lit_Latn': 'Lithuanian (written in Latin script)',
    'lug_Latn': 'Luganda (written in Latin script)',
    'lav_Latn': 'Latvian (written in Latin script)',
    'lzh_Hans': 'Literary Chinese (written in Simplified Chinese script)',
    'mai_Deva': 'Maithili (written in the Devanagari script)',
    'mlg_Latn': 'Malagasy (written in Latin script)',
    'mri_Latn': 'Maori (written in Latin script)',
    'mkd_Cyrl': 'Macedonian (written in the Cyrillic script)',
    'mal_Mlym': 'Malayalam (written in the Malayalam script)',
    'mon_Cyrl': 'Mongolian (written in the Cyrillic script)',
    'mon_Mong': 'Mongolian (written in the traditional Mongolian script)',
    'mni_Beng': 'Manipuri (written in the Bengali script)',
    'mar_Deva': 'Marathi (written in the Devanagari script)',
    'msa_Latn': 'Malay (written in Latin script)',
    'mlt_Latn': 'Maltese (written in Latin script)',
    'mww_Latn': 'Hmong Daw (written in Latin script)',
    'mya_Mymr': 'Myanmar (Burmese) (written in the Myanmar script)',
    'nob_Latn': 'Norwegian (written in Latin script)',
    'nep_Deva': 'Nepali (written in the Devanagari script)',
    'nld_Latn': 'Dutch (written in Latin script)',
    'nso_Latn': 'Northern Sotho (written in Latin script)',
    'nya_Latn': 'Chichewa (written in Latin script)',
    'ori_Orya': 'Odia (written in the Odia script)',
    'otq_Latn': 'Querétaro Otomi (written in Latin script)',
    'pan_Guru': 'Punjabi (written in the Gurmukhi script)',
    'pol_Latn': 'Polish (written in Latin script)',
    'prs_Arab': 'Dari (written in the Arabic script)',
    'pus_Arab': 'Pashto (written in the Arabic script)',
    'por_Latn': 'Portuguese (Brazil) (written in Latin script)',
    'por_Latn_PT': 'Portuguese (Portugal) (written in Latin script)',
    'ron_Latn': 'Romanian (written in Latin script)',
    'rus_Cyrl': 'Russian (written in the Cyrillic script)',
    'run_Latn': 'Kirundi (written in Latin script)',
    'kin_Latn': 'Kinyarwanda (written in Latin script)',
    'snd_Arab': 'Sindhi (written in the Arabic script)',
    'sin_Sinh': 'Sinhala (written in the Sinhala script)',
    'slk_Latn': 'Slovak (written in Latin script)',
    'slv_Latn': 'Slovenian (written in Latin script)',
    'smo_Latn': 'Samoan (written in Latin script)',
    'sna_Latn': 'Shona (written in Latin script)',
    'som_Latn': 'Somali (written in Latin script)',
    'sqi_Latn': 'Albanian (written in Latin script)',
    'srp_Cyrl': 'Serbian (Cyrillic) (written in the Cyrillic script)',
    'srp_Latn': 'Serbian (Latin) (written in Latin script)',
    'sot_Latn': 'Sesotho (written in Latin script)',
    'swe_Latn': 'Swedish (written in Latin script)',
    'swa_Latn': 'Swahili (written in Latin script)',
    'tam_Taml': 'Tamil (written in the Tamil script)',
    'tel_Telu': 'Telugu (written in the Telugu script)',
    'tha_Thai': 'Thai (written in the Thai script)',
    'tir_Ethi': 'Tigrinya (written in the Ethiopic script)',
    'tlh_Latn': 'Klingon (Latin) (written in Latin script)',
    'tlh_Piqd': 'Klingon (pIqaD) (written in the pIqaD script)',
    'tuk_Latn': 'Turkmen (written in Latin script)',
    'tsn_Latn': 'Tswana (written in Latin script)',
    'ton_Latn': 'Tongan (written in Latin script)',
    'tur_Latn': 'Turkish (written in Latin script)',
    'tat_Cyrl': 'Tatar (written in the Cyrillic script)',
    'tah_Latn': 'Tahitian (written in Latin script)',
    'uig_Arab': 'Uyghur (written in the Arabic script)',
    'ukr_Cyrl': 'Ukrainian (written in the Cyrillic script)',
    'urd_Arab': 'Urdu (written in the Arabic script)',
    'uzb_Latn': 'Uzbek (written in Latin script)',
    'vie_Latn': 'Vietnamese (written in Latin script)',
    'xho_Latn': 'Xhosa (written in Latin script)',
    'yor_Latn': 'Yoruba (written in Latin script)',
    'yua_Latn': 'Yucatec Maya (written in Latin script)',
    'yue_Hant': 'Cantonese (written in Traditional Chinese script)',
    'zho_Hans': 'Chinese (Simplified) (written in Simplified Chinese script)',
    'zho_Hant': 'Chinese (Traditional) (written in Traditional Chinese script)',
    'zul_Latn': 'Zulu (written in Latin script)',
}

def get_all_supported_languages():
    """
    Get all languages supported in our mapping.
    
    Returns:
        list: List of language codes in ISO 639-3_Script format
    """
    return list(MS_TO_ISO.values())

def iso_to_ms_code(iso_code):
    """
    Convert ISO 639-3_Script code to Microsoft language code.
    
    Args:
        iso_code (str): ISO code to convert
        
    Returns:
        str: Microsoft language code or None if not found
    """
    if iso_code in ISO_TO_MS:
        return ISO_TO_MS[iso_code]
    
    # Try to find by language code part
    iso_base = iso_code.split('_')[0]
    for iso, ms in ISO_TO_MS.items():
        if iso.startswith(iso_base + '_'):
            return ms
    
    return None

def ms_to_iso_code(ms_code):
    """
    Convert Microsoft language code to ISO 639-3_Script code.
    
    Args:
        ms_code (str): Microsoft code to convert
        
    Returns:
        str: ISO language code or None if not found
    """
    if ms_code in MS_TO_ISO:
        return MS_TO_ISO[ms_code]
    return None

def get_language_name(iso_code, include_script=False):
    """
    Get the language name for an ISO code.
    
    Args:
        iso_code (str): ISO code
        include_script (bool): Whether to include script information
        
    Returns:
        str: Language name or the code itself if not found
    """
    if include_script:
        return LANGUAGE_NAMES_WITH_SCRIPT.get(iso_code, iso_code)
    else:
        return LANGUAGE_NAMES.get(iso_code, iso_code)