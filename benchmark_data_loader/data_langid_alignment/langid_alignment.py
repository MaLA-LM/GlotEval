#!/usr/bin/env python3
import os
import csv
import re
import argparse
from difflib import SequenceMatcher
import unicodedata
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(grandparent_dir)

# Import iso639-lang package
from iso639 import Lang, is_language, iter_langs
from iso639.exceptions import InvalidLanguageValue, DeprecatedLanguageValue

# Import GlotScript
from GlotScript import get_script_predictor

# Import data loaders
from benchmark_data_loader.data_loader import (
    load_americasnlp_data,
    load_in22_data,
    load_flores_plus_data,
    load_flores200_data,
    load_nteu_data,
    load_ntrex128_data,
    load_tatoeba_data,
    load_aya_data,
    load_polywrite_data,
    load_mafand_data,
    load_pbc_data,
    load_mala_data,
    load_mmmlu_data,
    load_global_mmlu_data,
    load_sib200_data,
    load_wikiann_data,
    load_ud_data,
    load_taxi1500_data,
    load_xlsum_data,
    load_tico19_data
)

############################################################
## HELPERS: Language Label Parsing and Matching
############################################################

def normalize_text(text):
    """Normalize text by removing diacritics, lowercasing, and removing non-alphanumeric chars"""
    if not text:
        return ""
    # Normalize to form compatible with decomposition
    text = unicodedata.normalize('NFKD', text)
    # Remove diacritics (category "Mn" is "Mark, Nonspacing")
    text = ''.join([c for c in text if not unicodedata.category(c) == 'Mn'])
    # Lowercase and keep only alphanumeric
    text = ''.join([c.lower() for c in text if c.isalnum() or c.isspace()])
    return text

def parse_language_label(lang_label):
    """
    Parse a language label like "spa_Latn" into ("spa", "Latn").
    For labels like "Abaza-ATB", extract "Abaza" and "ATB".
    """
    if not lang_label:
        return None, None
    
    # Remove parentheses, brackets and clean whitespace
    label = re.sub(r'\([^)]*\)', '', lang_label)
    label = re.sub(r'\[[^\]]*\]', '', label)
    label = label.strip()
    
    # Pattern for standard ISO language code formats
    if "_" in label:
        iso_part, script_part = label.split("_", 1)
    elif "-" in label:
        # Check if it's an ISO code with region like "pt-BR"
        if len(label) == 5 and label[2] == "-" and label[0:2].isalpha() and label[3:5].isalpha():
            iso_part = label
            script_part = None
        else:
            # For UD treebank format like "Abaza-ATB"
            iso_part, script_part = label.split("-", 1)
    else:
        iso_part = label
        script_part = None
    
    return iso_part, script_part

def match_language_via_iso639(iso_part):
    """
    Attempt to match a language via iso639-lang package.
    Returns (lang_obj, status) where lang_obj is a Lang object if found,
    otherwise None. Status is a string describing the match result.
    """
    if not iso_part:
        return None, "no_iso_part"
    
    try:
        # Try direct match
        lang_obj = Lang(iso_part)
        return lang_obj, "exact_match"
    except InvalidLanguageValue:
        # Try fuzzy matching
        return fuzzy_match_language(iso_part)
    except DeprecatedLanguageValue as e:
        # Handle deprecated codes
        try:
            new_lang_obj = Lang(e.change_to)
            return new_lang_obj, "deprecated_code"
        except Exception:
            return None, "error_deprecated"
    except Exception:
        return None, "error_unknown"

def fuzzy_match_language(text):
    """
    Attempt to fuzzy match a language name against all available languages.
    Returns (lang_obj, status) where lang_obj is a Lang object if a good match is found,
    otherwise None. Status is a string describing the match result.
    """
    if not text:
        return None, "empty_text"
    
    # Normalize the input text
    norm_text = normalize_text(text)
    
    # First try exact matching with normalized form
    all_langs = list(iter_langs())
    
    # Create lookup dictionaries for faster searching
    name_to_lang = {normalize_text(lang.name): lang for lang in all_langs}
    iso3_to_lang = {lang.pt3: lang for lang in all_langs if lang.pt3}
    iso1_to_lang = {lang.pt1: lang for lang in all_langs if lang.pt1}
    
    # Try exact match on normalized name
    if norm_text in name_to_lang:
        return name_to_lang[norm_text], "norm_name_exact"
    
    # Try exact match on ISO codes
    if text in iso3_to_lang:
        return iso3_to_lang[text], "iso3_exact"
    
    if text in iso1_to_lang:
        return iso1_to_lang[text], "iso1_exact"
    
    # If no exact match, try fuzzy matching on names
    best_ratio = 0
    best_match = None
    
    # First try to see if it's a substring of any language name
    for lang in all_langs:
        if norm_text in normalize_text(lang.name):
            norm_lang_name = normalize_text(lang.name)
            ratio = len(norm_text) / len(norm_lang_name)
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = lang
    
    # If we found a good substring match (at least 60% of chars matching)
    if best_ratio > 0.6:
        return best_match, f"substring_match_{best_ratio:.2f}"
    
    # Otherwise try sequence matching for similarity
    best_ratio = 0
    best_match = None
    
    for lang in all_langs:
        norm_lang_name = normalize_text(lang.name)
        ratio = SequenceMatcher(None, norm_text, norm_lang_name).ratio()
        
        # Also check other names
        for other_name in lang.other_names():
            norm_other = normalize_text(other_name)
            other_ratio = SequenceMatcher(None, norm_text, norm_other).ratio()
            ratio = max(ratio, other_ratio)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = lang
    
    # If we have a reasonable match (at least 70% similarity)
    if best_ratio > 0.7:
        return best_match, f"fuzzy_match_{best_ratio:.2f}"
    
    # No good match found
    return None, "no_match"

############################################################
## Script detection with GlotScript
############################################################

script_predictor = get_script_predictor()

# CJK character sets for specific script detection
CJK_CHAR_SETS = {
    # Characters unique to Simplified Chinese
    "HANS_CHARS": set([
        '么', '书', '习', '乐', '争', '于', '亚', '亿', '从', '众',
        '会', '体', '余', '侠', '侣', '俩', '保', '儿', '克', '党',
        '册', '写', '军', '农', '冲', '决', '况', '净', '准', '凤',
        '処', '击', '制', '剂', '办', '务', '动', '劳', '势', '区',
        '单', '卖', '卫', '厂', '历', '压', '厅', '县', '叠', '只',
        '台', '后', '向', '吓', '启', '员', '团', '围', '国', '图',
        '圆', '块', '坏', '坚', '坛', '垒', '垦', '备', '复', '够',
        '头', '夹', '夺', '奋', '妈', '姐', '娘', '学', '宁', '宝',
        '实', '审', '对', '专', '导', '层', '岁', '岛', '币', '师',
        '带', '帮', '广', '庆', '应', '开', '张', '录', '归', '当',
        '录', '恒', '悬', '惊', '惠', '惯', '愿', '戏', '战', '户',
        '担', '拟', '择', '挤', '挥', '据', '损', '护', '报', '担',
        '拥', '挟', '择', '据', '捣', '掷', '摄', '摆', '摇', '斗',
        '断', '时', '显', '晓', '暂', '术', '机', '杀', '权', '来',
        '极', '构', '样', '档', '桥', '梦', '检', '楼', '权', '岁',
        '温', '满', '灭', '灯', '炼', '烟', '烦', '热', '爱', '爷',
        '牵', '犹', '状', '独', '猎', '猪', '玛', '环', '现', '确',
        '础', '签', '纤', '纪', '纲', '纳', '纵', '纸', '纹', '纺',
        '绍', '经', '给', '络', '统', '继', '绩', '维', '绿', '网',
        '罗', '罚', '罢', '义', '习', '职', '联', '胜', '肃', '肠',
        '肤', '肿', '胁', '脉', '脏', '脑', '脱', '腾', '舰', '艰',
        '艺', '节', '范', '荐', '荣', '药', '虑', '虚', '虫', '虽',
        '蚁', '蚂', '蚊', '蚀', '蜗', '蝇', '蝴', '蝶', '见', '观',
        '视', '览', '觉', '誉', '认', '说', '课', '谁', '调', '谅',
        '谈', '谋', '谓', '谢', '贝', '账', '贡', '财', '责', '败',
        '贯', '购', '贵', '贺', '资', '赌', '赏', '赐', '赛', '赶',
        '赵', '车', '转', '轮', '软', '轻', '载', '较', '辉', '农',
        '边', '达', '迁', '过', '运', '还', '这', '进', '连', '迟',
        '适', '选', '逻', '递', '远', '违', '铃', '铅', '银', '链',
        '锋', '锐', '错', '锡', '锦', '键', '镇', '长', '门', '闪',
        '闭', '问', '闲', '间', '闻', '阅', '队', '阳', '阴', '阶',
        '际', '陆', '陈', '险', '隐', '难', '雾', '靠', '韦', '页',
        '项', '须', '顺', '顾', '顿', '颖', '颗', '题', '额', '风',
        '飞', '饭', '饮', '饰', '饱', '饿', '馆', '马', '驱', '驳',
        '验', '骂', '鱼', '鲁', '鲜', '鸟', '鸡', '鸣', '鸭', '鸽',
        '鹅', '鹰', '黄', '鼻', '齐', '齿', '龙', '龚', '龟'
    ]),

    # Characters unique to Traditional Chinese
    "HANT_CHARS": set([
        '個', '為', '義', '並', '書', '來', '體', '處', '執', '學',
        '歷', '灣', '產', '畫', '異', '發', '著', '藝', '見', '觀',
        '計', '說', '讀', '變', '貓', '贊', '賣', '賽', '贏', '過',
        '達', '遊', '運', '進', '這', '道', '選', '邊', '開', '關',
        '陽', '際', '雙', '雞', '雪', '電', '須', '頭', '顔', '題',
        '風', '飛', '飯', '餓', '館', '馬', '體', '髮', '鬥', '鬼',
        '魚', '鳥', '鳳', '鷄', '麗', '麥', '麼', '黃', '點', '鼻',
        '齊', '齒', '龍', '龜', '閱', '僱', '儘', '啟', '囪', '媼',
        '嫻', '廁', '彙', '擊', '檯', '氳', '滷', '漿', '澀', '為',
        '痲', '癡', '皚', '矓', '秈', '糉', '綵', '膩', '臟', '葉',
        '藉', '藝', '蘋', '說', '讀', '變', '貓', '賣', '贊', '踴',
        '蹌', '軀', '醞', '鉅', '鍛', '鬚', '麩', '鼕', '齶'
    ]),

    # Japanese specific characters (hiragana, katakana, and specific kanji)
    "JPAN_CHARS": set([
        # Hiragana
        'あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ',
        'さ', 'し', 'す', 'せ', 'そ', 'た', 'ち', 'つ', 'て', 'と',
        'な', 'に', 'ぬ', 'ね', 'の', 'は', 'ひ', 'ふ', 'へ', 'ほ',
        'ま', 'み', 'む', 'め', 'も', 'や', 'ゆ', 'よ', 'ら', 'り',
        'る', 'れ', 'ろ', 'わ', 'を', 'ん', 'が', 'ぎ', 'ぐ', 'げ',
        'ご', 'ざ', 'じ', 'ず', 'ぜ', 'ぞ', 'だ', 'ぢ', 'づ', 'で',
        'ど', 'ば', 'び', 'ぶ', 'べ', 'ぼ', 'ぱ', 'ぴ', 'ぷ', 'ぺ',
        'ぽ', 'っ', 'ゃ', 'ゅ', 'ょ',
        # Katakana
        'ア', 'イ', 'ウ', 'エ', 'オ', 'カ', 'キ', 'ク', 'ケ', 'コ',
        'サ', 'シ', 'ス', 'セ', 'ソ', 'タ', 'チ', 'ツ', 'テ', 'ト',
        'ナ', 'ニ', 'ヌ', 'ネ', 'ノ', 'ハ', 'ヒ', 'フ', 'ヘ', 'ホ',
        'マ', 'ミ', 'ム', 'メ', 'モ', 'ヤ', 'ユ', 'ヨ', 'ラ', 'リ',
        'ル', 'レ', 'ロ', 'ワ', 'ヲ', 'ン', 'ガ', 'ギ', 'グ', 'ゲ',
        'ゴ', 'ザ', 'ジ', 'ズ', 'ゼ', 'ゾ', 'ダ', 'ヂ', 'ヅ', 'デ',
        'ド', 'バ', 'ビ', 'ブ', 'ベ', 'ボ', 'パ', 'ピ', 'プ', 'ペ',
        'ポ', 'ッ', 'ャ', 'ュ', 'ョ', 'ー',
        # Japan specific kanji
        '畠', '蘂', '鰯', '倶', '搔', '擧', '曉', '栗', '櫻', '毎',
        '洽', '涙', '渕', '溪', '漑', '灯', '犁', '祇', '禰', '突',
        '繋', '縣', '莖', '蔣', '訳', '謂', '賓', '遅', '醉', '釋',
        '鎌', '鑓', '靜', '顛', '驛', '髙', '髪', '鷗', '麴', '齋'
    ])
}

def detect_cjk_script(text):
    """
    Detect specific CJK script (Hans, Hant, Jpan) from text containing Hani characters.
    
    Args:
        text (str): Text sample potentially containing CJK characters
        
    Returns:
        str: Detected script code - 'Hans' (Simplified Chinese), 'Hant' (Traditional Chinese),
             'Jpan' (Japanese), or 'Hani' if undetermined
    """
    if not text:
        return "Hani"  # Default
    
    # Count characters by script
    hans_count = 0
    hant_count = 0
    jpan_count = 0
    
    for char in text:
        if char in CJK_CHAR_SETS["HANS_CHARS"]:
            hans_count += 1
        if char in CJK_CHAR_SETS["HANT_CHARS"]:
            hant_count += 1
        if char in CJK_CHAR_SETS["JPAN_CHARS"]:
            jpan_count += 1
    
    # If no specific characters detected, return generic Hani
    if hans_count == 0 and hant_count == 0 and jpan_count == 0:
        return "Hani"
    
    # Calculate proportions
    total = hans_count + hant_count + jpan_count
    hans_ratio = hans_count / total if total > 0 else 0
    hant_ratio = hant_count / total if total > 0 else 0
    jpan_ratio = jpan_count / total if total > 0 else 0
    
    # Determine script based on the highest proportion
    if jpan_ratio > 0.1:  # Even a small amount of Japanese-specific characters suggests Japanese
        return "Jpan"
    elif hans_ratio > hant_ratio:
        return "Hans"
    elif hant_ratio > hans_ratio:
        return "Hant"
    
    # Default fallback
    return "Hani"

def detect_script(text_samples):
    """
    Detect script from a list of text samples using GlotScript.
    Returns (script, confidence, details).
    """
    if not text_samples:
        return None, 0.0, {}
    
    # Count script predictions
    script_counts = {}
    total_samples = 0
    
    for text in text_samples:
        # Ensure text is a string
        if not text or not isinstance(text, str):
            continue
        
        try:
            result = script_predictor(text)
            if not result or not isinstance(result, tuple):
                continue
                
            top_script = result[0]
            top_conf = result[1]
            
            if not top_script:
                continue
            
            # Special handling for CJK scripts
            if top_script == "Hani":
                # Refine to Hans, Hant, or Jpan
                refined_script = detect_cjk_script(text)
                top_script = refined_script
            
            # Update script counts
            total_samples += 1
            if top_script not in script_counts:
                script_counts[top_script] = 0
            script_counts[top_script] += 1
            
        except Exception as e:
            logger.error(f"Error detecting script for text: {str(e)}")
            continue
    
    # If no valid predictions
    if not script_counts or total_samples == 0:
        return None, 0.0, {}
    
    # Get the most common script
    top_script = max(script_counts, key=script_counts.get)
    confidence = script_counts[top_script] / total_samples
    
    return top_script, confidence, {"counts": script_counts}

############################################################
## Benchmark dataset handling
############################################################

def guess_benchmark_from_filename(filename):
    """
    Guess which benchmark a file belongs to based on its name.
    """
    base = os.path.splitext(filename)[0]
    if "_langs" in base:
        bench = base.split("_langs")[0]
    else:
        bench = base
    return bench

# Dictionary mapping benchmark names to their data loading functions
BENCHMARK_LOADERS = {
    "americasnlp": load_americasnlp_data,
    "in22": load_in22_data,
    "flores_plus": load_flores_plus_data,
    "flores200": load_flores200_data,
    "nteu": load_nteu_data,
    "ntrex128": load_ntrex128_data,
    "tatoeba": load_tatoeba_data,
    "aya": load_aya_data,
    "polywrite": load_polywrite_data,
    "mafand": load_mafand_data,
    "pbc": load_pbc_data,
    "mala": load_mala_data,
    "mmmlu": load_mmmlu_data,
    "global_mmlu": load_global_mmlu_data,
    "sib200": load_sib200_data,
    "wikiann": load_wikiann_data,
    "ud": load_ud_data,
    "taxi1500": load_taxi1500_data,
    "xlsum": load_xlsum_data,
    "tico19": load_tico19_data
}

def load_sample_lines(benchmark_name, lang_label, max_lines=3):
    """
    Load sample lines from the specified benchmark for a given language.
    Returns sample text lines and error message if any.
    """
    # Default fallback
    lines = ["Placeholder text for detection."]
    error_msg = None
    
    try:
        lang_label = lang_label.strip()
        iso_part, script_part = parse_language_label(lang_label)
        
        # Check if benchmark is supported
        if benchmark_name not in BENCHMARK_LOADERS:
            error_msg = f"No known loader for benchmark: {benchmark_name}"
            logger.warning(error_msg)
            return lines, error_msg
        
        loader_func = BENCHMARK_LOADERS[benchmark_name]
        
        # Special handling for different benchmark types
        try:
            # Translation benchmarks (need source and target language)
            if benchmark_name in ["americasnlp", "in22", "flores_plus", "flores200",
                                  "nteu", "ntrex128", "tatoeba", "mafand", "tico19"]:
                # Determine source and target languages
                if benchmark_name == "americasnlp":
                    if lang_label != "spa_Latn":
                        src_lang, tgt_lang = lang_label, "spa_Latn"
                    else:
                        src_lang, tgt_lang = "agr_Latn", "spa_Latn"
                elif benchmark_name in ["in22", "flores_plus", "flores200"]:
                    if lang_label != "eng_Latn":
                        src_lang, tgt_lang = lang_label, "eng_Latn"
                    else:
                        src_lang, tgt_lang = "ces_Latn", "eng_Latn"
                elif benchmark_name == "nteu":
                    if lang_label != "en":
                        src_lang, tgt_lang = lang_label, "en"
                    else:
                        src_lang, tgt_lang = "es", "en"
                elif benchmark_name == "ntrex128":
                    if lang_label != "eng":
                        src_lang, tgt_lang = lang_label, "eng"
                    else:
                        src_lang, tgt_lang = "fra", "eng"
                elif benchmark_name == "tatoeba":
                    if lang_label != "eng":
                        src_lang, tgt_lang = lang_label, "eng"
                    else:
                        src_lang, tgt_lang = "spa", "eng"
                elif benchmark_name == "mafand":
                    if lang_label in ["en", "fr"]:
                        src_lang, tgt_lang = lang_label, "hau"
                    else:
                        src_lang, tgt_lang = lang_label, "en"
                elif benchmark_name == "tico19":
                    if lang_label != "en":
                        src_lang, tgt_lang = lang_label, "en"
                    else:
                        src_lang, tgt_lang = "fa", "en"
                
                # Load data
                src_texts, _ = loader_func(src_lang, tgt_lang, split="test", limit_samples=max_lines)
                lines = src_texts if src_texts else lines
            
            # Single language text benchmarks
            elif benchmark_name in ["aya", "polywrite"]:
                all_texts = loader_func(lang_label)
                lines = all_texts[:max_lines] if all_texts else lines
            
            # Text classification benchmarks
            elif benchmark_name in ["mmmlu", "global_mmlu"]:
                data = loader_func(lang_label, split="test", limit_samples=max_lines)
                lines = [ex["question"] for ex in data] if data else ["No data?"]
            
            # Token classification benchmarks
            elif benchmark_name in ["wikiann"]:
                data = loader_func(lang_label, split="test", limit_samples=max_lines)
                lines = [" ".join(ex.get("tokens", [])) for ex in data[:max_lines]] if data else lines
            
            # Tree bank data (UD)
            elif benchmark_name == "ud":
                data = loader_func(lang_label, split="test", limit_samples=max_lines)
                lines = [" ".join(ex.get("tokens", [])) for ex in data[:max_lines]] if data else lines
            
            # Other benchmark types
            elif benchmark_name in ["sib200", "taxi1500"]:
                data = loader_func(lang_label, split="test")
                lines = [d["text"] for d in data[:max_lines]] if data else lines
            
            elif benchmark_name == "xlsum":
                data_src, _ = loader_func(lang_label, split="test", limit_samples=max_lines)
                lines = data_src if data_src else lines
            
            # Language modeling
            elif benchmark_name in ["pbc", "mala"]:
                split = "test" if benchmark_name == "pbc" else "validation"
                texts = loader_func(lang_label, split=split)
                lines = texts[:max_lines] if texts else lines
                
        except Exception as e:
            error_msg = f"Error loading {benchmark_name} data for {lang_label}: {str(e)}"
            logger.error(error_msg)

    except Exception as e:
        error_msg = f"General error loading data for {lang_label} in {benchmark_name}: {str(e)}"
        logger.error(error_msg)

    return lines, error_msg

############################################################
## Main processing function
############################################################

def generate_aligned_key(orig_label, iso3_code, script_code, match_status, script_conf):
    """
    Generate a key for aligned file in the format "original | iso3_script".
    If uncertain, add "?" to the key.
    
    Args:
        orig_label (str): Original language label
        iso3_code (str): ISO639-3 code
        script_code (str): Script code
        match_status (str): Status of language matching
        script_conf (float): Confidence of script detection
        
    Returns:
        str: Generated key for aligned file
    """
    # Start with original label
    key = orig_label
    
    # Check if we have valid iso3 and script codes
    if iso3_code and script_code:
        # Handle special cases for script codes
        if script_code == "Hani":
            # Default Hani script to Hans for Chinese languages if it's a close match
            if iso3_code == "zho" or iso3_code == "cmn":
                script_code = "Hans"
        
        # Create the aligned part
        aligned_part = f"{iso3_code}_{script_code}"
        
        # Determine if we need to add a question mark
        is_uncertain = False
        
        # Check language match confidence
        if not match_status.startswith("exact_match") and not match_status.startswith("iso3_exact"):
            is_uncertain = True
            
        # Check script detection confidence
        try:
            if float(script_conf) < 1.0:
                is_uncertain = True
        except (ValueError, TypeError):
            is_uncertain = True
            
        # Add question mark if uncertain
        if is_uncertain:
            aligned_part += "?"
            
        # Create the full key
        key = f"{orig_label} | {aligned_part}"
    else:
        # No valid mapping, add question mark
        key = f"{orig_label} | ?"
        
    return key

def process_language_config(config_path, max_lines=3, generate_aligned=True):
    """
    Process a language config file, performing language and script detection.
    If generate_aligned is True, also generates an aligned file with format "orig | iso_script"
    """
    filename = os.path.basename(config_path)
    benchmark_name = guess_benchmark_from_filename(filename)
    
    # Create output directory if it doesn't exist
    out_dir = os.path.join(os.path.dirname(config_path), "alignment_reports")
    os.makedirs(out_dir, exist_ok=True)
    
    # Create aligned directory if needed
    if generate_aligned:
        aligned_dir = os.path.join(os.path.dirname(config_path), "data_langid_aligned")
        os.makedirs(aligned_dir, exist_ok=True)
        aligned_file_path = os.path.join(aligned_dir, filename)
    
    out_csv_name = os.path.splitext(filename)[0] + "_script_report.csv"
    out_csv_path = os.path.join(out_dir, out_csv_name)
    
    # For tracking uncertain matches
    uncertain_matches = []
    
    logger.info(f"Processing {filename} => {out_csv_name}, benchmark={benchmark_name}")
    
    fieldnames = [
        "original_label",
        "parsed_lang_part",
        "parsed_script_part",
        "iso639_match_status",
        "iso639_pt1",
        "iso639_pt2b",
        "iso639_pt2t",
        "iso639_pt3",
        "iso639_name",
        "macro_status",
        "macro_language",
        "detected_script",
        "script_confidence",
        "sample_text",
        "data_load_error",
        "aligned_key"
    ]
    
    with open(config_path, "r", encoding="utf-8") as fin, \
         open(out_csv_path, "w", newline="", encoding="utf-8") as fout:
        
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        
        # For aligned file if requested
        aligned_lines = []
        
        lines = fin.read().splitlines()
        for line_idx, raw_line in enumerate(lines):
            label_str = raw_line.strip()
            if not label_str or label_str.startswith('#'):
                # Skip empty lines and comments
                if generate_aligned and label_str.startswith('#'):
                    aligned_lines.append(label_str)  # Preserve comments in aligned file
                continue
            
            logger.info(f"Processing language {line_idx+1}/{len(lines)}: {label_str}")
            
            try:
                # Parse language label
                iso_part, script_part = parse_language_label(label_str)
                
                # Match language using iso639-lang
                lang_obj, match_status = match_language_via_iso639(iso_part)
                
                # Get language information
                iso1 = lang_obj.pt1 if lang_obj else ""
                iso2b = lang_obj.pt2b if lang_obj else ""
                iso2t = lang_obj.pt2t if lang_obj else ""
                iso3 = lang_obj.pt3 if lang_obj else ""
                name = lang_obj.name if lang_obj else ""
                
                # Check if it's a macrolanguage or has a macrolanguage
                macro_status = ""
                macro_language = ""
                if lang_obj:
                    try:
                        scope = lang_obj.scope()
                        if scope == "Macrolanguage":
                            macro_status = "is_macro"
                        else:
                            # Check if it belongs to a macrolanguage
                            try:
                                macro = lang_obj.macro()
                                if macro:
                                    macro_status = "has_macro"
                                    macro_language = macro.name
                            except Exception:
                                pass
                    except Exception:
                        pass
                
                # Load sample lines from dataset
                sample_lines, error_msg = load_sample_lines(benchmark_name, label_str, max_lines=max_lines)
                
                # Validate sample lines are strings
                valid_samples = []
                for text in sample_lines:
                    if isinstance(text, str):
                        valid_samples.append(text)
                    else:
                        logger.warning(f"Non-string sample found for {label_str}: {type(text)}")
                
                # Use sample text for script detection if available
                script = ""
                script_conf = 0.0
                if valid_samples and valid_samples[0] != "Placeholder text for detection.":
                    try:
                        script, script_conf, _ = detect_script(valid_samples)
                    except Exception as e:
                        logger.error(f"Error in script detection for {label_str}: {str(e)}")
                
                # Get a sample of the text for reference
                sample_text = valid_samples[0] if valid_samples else ""
                if isinstance(sample_text, str) and len(sample_text) > 100:
                    sample_text = sample_text[:100] + "..."
                elif not isinstance(sample_text, str):
                    sample_text = f"[Non-string sample: {type(sample_text)}]"
                
                # Generate aligned key
                aligned_key = generate_aligned_key(
                    label_str, iso3, script, match_status, script_conf
                )
                
                # Check if uncertain and should be tracked
                if "?" in aligned_key:
                    uncertain_matches.append({
                        "benchmark": benchmark_name,
                        "original_label": label_str,
                        "iso639_match_status": match_status,
                        "iso639_pt3": iso3,
                        "iso639_name": name,
                        "detected_script": script,
                        "script_confidence": script_conf,
                        "aligned_key": aligned_key
                    })
                
                # Add aligned key to aligned lines if it's certain
                if generate_aligned and "?" not in aligned_key:
                    aligned_lines.append(aligned_key)
                elif generate_aligned:
                    # Still add uncertain keys but mark them
                    aligned_lines.append(f"# UNCERTAIN: {aligned_key}")
                
                row = {
                    "original_label": label_str,
                    "parsed_lang_part": iso_part if iso_part else "",
                    "parsed_script_part": script_part if script_part else "",
                    "iso639_match_status": match_status,
                    "iso639_pt1": iso1,
                    "iso639_pt2b": iso2b,
                    "iso639_pt2t": iso2t,
                    "iso639_pt3": iso3,
                    "iso639_name": name,
                    "macro_status": macro_status,
                    "macro_language": macro_language,
                    "detected_script": script if script else "",
                    "script_confidence": f"{script_conf:.2f}" if script_conf else "0.00",
                    "sample_text": sample_text,
                    "data_load_error": error_msg if error_msg else "",
                    "aligned_key": aligned_key
                }
                
                writer.writerow(row)
                
            except Exception as e:
                logger.error(f"Error processing {label_str}: {str(e)}")
                # Write a minimal row with error information
                error_key = f"{label_str} | ?"
                if generate_aligned:
                    aligned_lines.append(f"# ERROR: {error_key}")
                    
                try:
                    writer.writerow({
                        "original_label": label_str,
                        "data_load_error": f"Processing error: {str(e)}",
                        "parsed_lang_part": "",
                        "parsed_script_part": "",
                        "iso639_match_status": "",
                        "iso639_pt1": "",
                        "iso639_pt2b": "",
                        "iso639_pt2t": "",
                        "iso639_pt3": "",
                        "iso639_name": "",
                        "macro_status": "",
                        "macro_language": "",
                        "detected_script": "",
                        "script_confidence": "0.00",
                        "sample_text": "",
                        "aligned_key": error_key
                    })
                except Exception as write_err:
                    logger.error(f"Error writing error row: {str(write_err)}")
    
        # Write aligned file if requested
        if generate_aligned and aligned_lines:
            try:
                with open(aligned_file_path, "w", encoding="utf-8") as afile:
                    for line in aligned_lines:
                        afile.write(f"{line}\n")
                logger.info(f"Written aligned file: {aligned_file_path}")
            except Exception as e:
                logger.error(f"Error writing aligned file: {str(e)}")
    
    # Create a summary file of uncertain matches
    if uncertain_matches:
        uncertain_csv_path = os.path.join(out_dir, "uncertain_matches.csv")
        try:
            with open(uncertain_csv_path, "a", newline="", encoding="utf-8") as unc_file:
                fieldnames = [
                    "benchmark", "original_label", "iso639_match_status", 
                    "iso639_pt3", "iso639_name", "detected_script", 
                    "script_confidence", "aligned_key"
                ]
                
                # Check if file exists and is empty to determine if we need headers
                is_new_file = os.path.getsize(uncertain_csv_path) == 0 if os.path.exists(uncertain_csv_path) else True
                
                writer = csv.DictWriter(unc_file, fieldnames=fieldnames)
                if is_new_file:
                    writer.writeheader()
                    
                for match in uncertain_matches:
                    writer.writerow(match)
                    
            logger.info(f"Added {len(uncertain_matches)} uncertain matches to {uncertain_csv_path}")
        except Exception as e:
            logger.error(f"Error writing uncertain matches file: {str(e)}")
    
    logger.info(f"Done => {out_csv_path}")
    return out_csv_path

def main():
    parser = argparse.ArgumentParser(description="Language ID and Script Alignment Tool")
    parser.add_argument("--data_config_dir", default="benchmark_data_loader/data_langid_alignment", 
                        help="Directory containing language config files")
    parser.add_argument("--max_lines", type=int, default=3, 
                        help="Maximum number of sample lines to load for script detection")
    parser.add_argument("--file", type=str, help="Process a specific file only")
    parser.add_argument("--no-aligned", action="store_true", 
                        help="Skip generation of aligned files")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    data_config_dir = args.data_config_dir
    generate_aligned = not args.no_aligned
    
    if args.file:
        # Process a single file
        if not os.path.isfile(args.file):
            full_path = os.path.join(data_config_dir, args.file)
            if os.path.isfile(full_path):
                process_language_config(full_path, max_lines=args.max_lines, generate_aligned=generate_aligned)
            else:
                logger.error(f"File not found: {args.file}")
        else:
            process_language_config(args.file, max_lines=args.max_lines, generate_aligned=generate_aligned)
    else:
        # Process all text files in the directory
        processed_files = []
        for filename in os.listdir(data_config_dir):
            if filename.endswith("_langs.txt") and not filename.startswith('.'):
                filepath = os.path.join(data_config_dir, filename)
                processed_path = process_language_config(filepath, max_lines=args.max_lines, generate_aligned=generate_aligned)
                processed_files.append(processed_path)
        
        logger.info(f"Processed {len(processed_files)} files.")
        
        # Summarize uncertain matches
        uncertain_csv_path = os.path.join(data_config_dir, "alignment_reports", "uncertain_matches.csv")
        if os.path.exists(uncertain_csv_path):
            try:
                with open(uncertain_csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    uncertain_count = sum(1 for _ in reader)
                logger.info(f"\nFound {uncertain_count} uncertain language matches.")
                logger.info(f"See details in: {uncertain_csv_path}")
            except Exception as e:
                logger.error(f"Error reading uncertain matches summary: {str(e)}")

if __name__ == "__main__":
    main()