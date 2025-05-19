# filepath: d:\SemanticSearch\Tool\OIE.py
import os
import hashlib
import json
import re
import time
import traceback
import atexit
from dotenv import load_dotenv
from openie import StanfordOpenIE
import diskcache

# --- Global Constants and Configuration ---
CORENLP_DEFAULT_PATH = r'D:/SemanticSearch/CoreNLP'
OIE_CACHE_DIR = './oie_cache'

# Load environment variables (e.g., CORENLP_HOME if set elsewhere)
load_dotenv()

# Initialize Cache
try:
    OIE_CACHE = diskcache.Cache(OIE_CACHE_DIR)
    print(f"[INFO] DiskCache initialized at: {OIE_CACHE_DIR}")
except Exception as e:
    print(f"[ERROR] Failed to initialize DiskCache at {OIE_CACHE_DIR}: {e}")
    OIE_CACHE = None # Fallback if cache initialization fails

# Stanford OpenIE Client Properties
STANDARD_PROPERTIES = {
    'annotators': 'tokenize,ssplit,pos,lemma,ner,depparse,coref,natlog,openie',
    'openie.affinity_probability_cap': 0.8, # Higher precision
    'openie.triple.strict': False, # Allow non-strict triples
    'openie.max_entailments_per_clause': 3,
    'openie.resolve_coref': True, # Crucial for multi-sentence context
    'outputFormat': 'json',
    'timeout': 180000  # 3 minutes for potentially complex paragraphs
}

ENHANCED_PROPERTIES = {
    'annotators': 'tokenize,ssplit,pos,lemma,ner,depparse,coref,natlog,openie',
    'openie.affinity_probability_cap': 0.4, # Lowered for more recall (was 0.6)
    'openie.triple.strict': False,
    'openie.max_entailments_per_clause': 10, # Increased for deeper search (was 5)
    'openie.resolve_coref': True,
    'openie.min_relation_length': 1, # Allow shorter relations
    'openie.exhaustive': True, # Added for more exhaustive search
    'openie.ignore_affinity': True, # Added: Ignore affinity scores
    'openie.triple.all_nominals': True, # Added: Extract from nominalizations
    'outputFormat': 'json',
    'timeout': 300000  # 5 minutes (increased slightly due to more exhaustive options)
}

# Global Client Instances
_CLIENT_STANDARD = None
_CLIENT_ENHANCED = None

# --- Helper Functions ---
def _ensure_corenlp_home():
    """Ensures CORENLP_HOME is set, using a default path if necessary."""
    if 'CORENLP_HOME' not in os.environ:
        if os.path.exists(CORENLP_DEFAULT_PATH):
            os.environ['CORENLP_HOME'] = CORENLP_DEFAULT_PATH
            print(f"[INFO] CORENLP_HOME was not set. Using default: '{CORENLP_DEFAULT_PATH}'")
        else:
            print(f"[ERROR] CORENLP_HOME is not set and default path '{CORENLP_DEFAULT_PATH}' does not exist. StanfordOpenIE may not work.")
            return False
    # print(f"[INFO] CORENLP_HOME is set to: {os.environ['CORENLP_HOME']}")
    return True

# --- Client Management ---
def get_stanford_oie_client(use_enhanced_settings: bool, custom_properties: dict = None):
    """
    Gets or initializes a StanfordOpenIE client.
    Manages separate clients for standard and enhanced properties.
    """
    global _CLIENT_STANDARD, _CLIENT_ENHANCED

    if not _ensure_corenlp_home():
        return None

    # Determine which global client variable to use/update
    if custom_properties: # If custom properties are given, don't use shared global clients or create a new one based on settings
        client_to_use = None # Force new client for custom properties, not caching it globally
        print("[INFO] Using custom properties; a new client instance will be created if necessary but not stored globally.")
    elif use_enhanced_settings:
        client_to_use = _CLIENT_ENHANCED
    else:
        client_to_use = _CLIENT_STANDARD
    
    if client_to_use is None:
        props_to_use = custom_properties
        if props_to_use is None:
            props_to_use = ENHANCED_PROPERTIES if use_enhanced_settings else STANDARD_PROPERTIES
        
        # Ensure outputFormat is json, as the parsing logic relies on it.
        props_to_use['outputFormat'] = 'json'
        # Ensure coref is resolved for paragraph-level understanding (it's already in defaults but good to be explicit)
        props_to_use['openie.resolve_coref'] = True


        print(f"[INFO] Initializing StanfordOpenIE client (enhanced={use_enhanced_settings}, custom={custom_properties is not None}). This may take a moment...")
        print(f"[INFO] Using properties: {props_to_use}")
        try:
            # Suppress excessive logging from underlying libraries if possible (example)
            # import logging
            # logging.getLogger('stanfordnlp.server').setLevel(logging.WARNING)
            
            client_to_use = StanfordOpenIE(properties=props_to_use)
            if custom_properties:
                # Do not store clients with custom_properties globally
                pass
            elif use_enhanced_settings:
                _CLIENT_ENHANCED = client_to_use
            else:
                _CLIENT_STANDARD = client_to_use
            print(f"[INFO] StanfordOpenIE client (enhanced={use_enhanced_settings}, custom={custom_properties is not None}) initialized successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize StanfordOpenIE client (enhanced={use_enhanced_settings}): {e}")
            print(traceback.format_exc())
            return None
            
    return client_to_use

def close_oie_clients():
    """Closes any active StanfordOpenIE clients by releasing them."""
    global _CLIENT_STANDARD, _CLIENT_ENHANCED # Ensure we're modifying the globals
    clients_released_count = 0
    
    # Save the initial state of CORENLP_HOME to handle potential deletion by one client's __del__
    # before another client's __del__ is called.
    original_corenlp_home_value = os.environ.get('CORENLP_HOME')

    if _CLIENT_STANDARD:
        try:
            print("[INFO] Releasing StanfordOpenIE standard client...")
            # Setting to None should trigger __del__ if no other references,
            # which handles stopping the server via the client's __del__ method.
            _CLIENT_STANDARD = None # This might trigger __del__ which deletes CORENLP_HOME
            print("[INFO] StanfordOpenIE standard client released.")
            clients_released_count +=1
        except Exception as e:
            print(f"[ERROR] Error releasing standard client: {e}") 
    if _CLIENT_ENHANCED:
        try:
            # If CORENLP_HOME was present at the start of this function, but is now missing,
            # and the enhanced client is about to be released, temporarily restore CORENLP_HOME.
            if 'CORENLP_HOME' not in os.environ and original_corenlp_home_value is not None:
                print(f"[DEBUG] Temporarily restoring CORENLP_HOME='{original_corenlp_home_value}' for enhanced client's __del__.")
                os.environ['CORENLP_HOME'] = original_corenlp_home_value
            
            print("[INFO] Releasing StanfordOpenIE enhanced client...")
            _CLIENT_ENHANCED = None # This might trigger __del__ which deletes CORENLP_HOME
            print("[INFO] StanfordOpenIE enhanced client released.")
            clients_released_count +=1
        except Exception as e:
            print(f"[ERROR] Error releasing enhanced client: {e}")

    if clients_released_count == 0:
        print("[INFO] No active global OpenIE clients to release (they might have been custom instances or already released).")

# Register client cleanup at exit
atexit.register(close_oie_clients)

# --- Core Extraction and Caching ---
def _get_raw_triples_from_oie(text_block: str, client: StanfordOpenIE, properties_for_annotate: dict) -> list:
    """
    Internal function to extract raw triples from a text block using a given client and properties.
    Handles caching of results.
    """
    if not text_block or not client:
        return []

    # --- Caching Logic ---
    cache_key = None # Initialize cache_key
    if OIE_CACHE:
        cache_key_parts = [text_block, json.dumps(properties_for_annotate, sort_keys=True)]
        cache_key = hashlib.md5("||".join(cache_key_parts).encode('utf-8')).hexdigest()
        print(f"[DEBUG] Attempting to get from cache with key: {cache_key}") # Added
        cached_result = OIE_CACHE.get(cache_key)
        if cached_result is not None:
            print(f"[DEBUG] OIE Cache hit for key: {cache_key}. Returning {len(cached_result)} cached triples.") # Modified
            return cached_result
        else: # Added
            print(f"[DEBUG] OIE Cache miss for key: {cache_key}") # Added
    # --- End Caching Logic ---

    triples = []
    annotated_output_dict = None # Initialize to ensure it's available for logging, and clarify it's a dict
    try:
        print(f"[DEBUG] Annotating text with OpenIE (length: {len(text_block)} chars, first 100: '{text_block[:100]}...')") # Modified
        print(f"[DEBUG] Using properties for annotate: {json.dumps(properties_for_annotate)}") # Added
        
        annotation_start_time = time.time() # Added
        # CRITICAL CHANGE:
        # client.client.annotate with output_format='json' returns a Python dictionary directly.
        annotated_output_dict = client.client.annotate(text_block, properties=properties_for_annotate, output_format='json')
        annotation_end_time = time.time() # Added
        print(f"[DEBUG] Annotation call completed in {annotation_end_time - annotation_start_time:.2f} seconds.") # Added
        
        print(f"[DEBUG] Type of annotated_output_dict: {type(annotated_output_dict)}")
        # annotated_output_dict is now a Python dictionary.
        print(f"[DEBUG] Raw annotated_output_dict (first 500 chars): {str(annotated_output_dict)[:500]}")

        result_json = annotated_output_dict # Assign the dict directly
        # No longer need json.loads as client.client.annotate with output_format='json' returns a dict.
        # print("[DEBUG] Successfully parsed annotated_output_str as JSON.") # Removed this misleading print

        if not isinstance(result_json, dict):
            print(f"[ERROR] Expected a dict from annotation, but got {type(result_json)}. Content: {str(result_json)[:1000]}")
            # Do not cache on error (this path implies an unexpected error)
            return []

        if 'sentences' in result_json:
            print(f"[DEBUG] 'sentences' key found in JSON. Number of sentences: {len(result_json['sentences'])}") # Added
            for i, sentence_data in enumerate(result_json['sentences']):
                # Log detailed information for each sentence
                print(f"--- Detailed info for Sentence {i} ---")
                print(f"[DEBUG] Sentence {i} - Text (from tokens): {{ ' '.join([t['originalText'] for t in sentence_data.get('tokens', [])]) }}")
                print(f"[DEBUG] Sentence {i} - Tokens: {sentence_data.get('tokens')}")
                print(f"[DEBUG] Sentence {i} - Basic Dependencies: {sentence_data.get('basicDependencies')}")
                # print(f"[DEBUG] Sentence {i} - Enhanced Dependencies: {sentence_data.get('enhancedDependencies')}") # Potentially very verbose
                # print(f"[DEBUG] Sentence {i} - Parse tree: {sentence_data.get('parse')}") # Potentially very verbose
                
                if 'openie' in sentence_data:
                    print(f"[DEBUG] Sentence {i}: 'openie' key found. Number of raw triples: {len(sentence_data['openie'])}") # Added
                    print(f"[DEBUG] Sentence {i} raw OIE triples: {sentence_data['openie']}") 
                    for triple_data in sentence_data['openie']:
                        if all(k in triple_data for k in ['subject', 'relation', 'object']):
                            triples.append({
                                'subject': triple_data['subject'],
                                'relation': triple_data['relation'],
                                'object': triple_data['object']
                            })
                        else:
                            print(f"[DEBUG] Sentence {i}: Incomplete triple data: {triple_data}") # Modified
                else:
                     print(f"[DEBUG] Sentence {i}: No 'openie' key in sentence data. Keys present: {list(sentence_data.keys())}") # Modified
        else:
            print(f"[WARNING] No 'sentences' key in OpenIE output for text: {text_block[:100]}...")
            print(f"[DEBUG] Full JSON output structure (keys): {list(result_json.keys()) if isinstance(result_json, dict) else 'Not a dict'}") # Added


        if OIE_CACHE and cache_key: # Ensure cache_key is defined
            print(f"[DEBUG] Setting cache for key: {cache_key} with {len(triples)} triples.") # Added
            OIE_CACHE.set(cache_key, triples)
        return triples

    except json.JSONDecodeError as je: # This exception should ideally not be hit by our code's logic anymore.
                                       # If client.client.annotate itself has an issue decoding, it might raise AnnotationException.
        print(f"[ERROR] JSONDecodeError parsing OpenIE output (this might indicate an issue within the CoreNLP client or an unexpected string response): {je}")
        error_output_to_log = str(annotated_output_dict) if annotated_output_dict is not None else "No output captured before error"
        print(f"[ERROR] Received output that might be related (first 500 chars): {error_output_to_log[:500]}")
        # Do not cache on JSON decode error
        return []
    except Exception as e:
        print(f"[ERROR] Exception during OpenIE triple extraction: {e}")
        print(traceback.format_exc())
        error_output_to_log = str(annotated_output_dict) if annotated_output_dict is not None else "No output captured before error"
        print(f"[DEBUG] State of 'annotated_output_dict' at time of exception: {error_output_to_log[:500]}")
        # Do not cache on other critical errors
        return []

# --- Filtering Functions ---
def filter_low_quality_triples(triples: list, min_subject_len_words: int = 1,
                               min_relation_len_words: int = 1, min_object_len_words: int = 1) -> list:
    """Filters triples based on length and common heuristic issues."""
    filtered = []
    if not triples:
        return []

    for triple in triples:
        s = triple['subject'].strip()
        r = triple['relation'].strip()
        o = triple['object'].strip()

        if not s or not r or not o: # Skip if any part is empty after stripping
            continue

        # Word count check
        if (len(s.split()) < min_subject_len_words or
            len(r.split()) < min_relation_len_words or
            len(o.split()) < min_object_len_words):
            continue
        
        # Avoid overly generic relations or subjects/objects (example heuristic)
        if r.lower() in ["is", "are", "was", "were", "be", "has", "have", "had"]:
             if len(s.split()) < 2 and len(o.split()) < 2: # e.g. "He is tall" is fine, "It is it" is not
                 pass # Allow simple S-V-Adj or S-V-N, but could be stricter

        # Common problematic patterns in relation
        if any(bad_pattern in r.lower() for bad_pattern in ['can can', 'are are', 'is is', 'of are']):
            continue
        
        # Check for unbalanced parentheses (simple check)
        if s.count('(') != s.count(')') or \
           r.count('(') != r.count(')') or \
           o.count('(') != o.count(')'):
            continue
        
        # Avoid triples that are just pronouns or very short
        if len(s) < 2 or len(r) < 2 or len(o) < 2 : # Character length
             if not (len(s.split()) > 1 or len(r.split()) > 1 or len(o.split()) > 1): # Unless multi-word
                continue


        filtered.append(triple)
    return filtered

def filter_duplicate_triples(triples: list) -> list:
    """Removes duplicate triples based on normalized string representations."""
    if not triples:
        return []
        
    unique_triples = []
    seen_normalized = set()
    
    for triple in triples:
        # Normalize by lowercasing, stripping, and collapsing multiple spaces
        norm_s = re.sub(r'\s+', ' ', triple['subject'].lower().strip())
        norm_r = re.sub(r'\s+', ' ', triple['relation'].lower().strip())
        norm_o = re.sub(r'\s+', ' ', triple['object'].lower().strip())
        
        normalized_tuple = (norm_s, norm_r, norm_o)
        
        if normalized_tuple not in seen_normalized:
            seen_normalized.add(normalized_tuple)
            unique_triples.append(triple)
            
    return unique_triples

# --- Main Public Function ---
def extract_relations_from_paragraph(paragraph_text: str, use_enhanced_settings: bool = True,
                                     custom_properties: dict = None) -> list:
    """
    Extracts, filters, and deduplicates OpenIE relations from a paragraph.

    Args:
        paragraph_text: The paragraph text to process.
        use_enhanced_settings: If True, uses ENHANCED_PROPERTIES, else STANDARD_PROPERTIES.
                               Ignored if custom_properties are provided.
        custom_properties: Optional dictionary to override default OpenIE properties.

    Returns:
        A list of dictionaries, where each dictionary represents a filtered triple.
    """
    if not paragraph_text.strip():
        print("[WARNING] Empty paragraph text provided.")
        return []

    start_time = time.time()
    print(f"\n[INFO] Starting relation extraction for paragraph (enhanced={use_enhanced_settings}, custom_props={'Yes' if custom_properties else 'No'}).")
    
    oie_client = get_stanford_oie_client(use_enhanced_settings, custom_properties)
    if not oie_client:
        print("[ERROR] Failed to get OpenIE client. Aborting extraction.")
        return []

    # Determine properties used for this extraction (for caching key and logging)
    current_props = custom_properties
    if current_props is None:
        current_props = ENHANCED_PROPERTIES if use_enhanced_settings else STANDARD_PROPERTIES

    raw_triples = _get_raw_triples_from_oie(paragraph_text, oie_client, current_props)
    print(f"[DEBUG] Raw triples extracted: {len(raw_triples)}") # Uncommented

    filtered_triples = filter_low_quality_triples(raw_triples)
    print(f"[DEBUG] Triples after quality filtering: {len(filtered_triples)}") # Uncommented

    final_triples = filter_duplicate_triples(filtered_triples)
    print(f"[DEBUG] Triples after duplicate filtering: {len(final_triples)}") # Uncommented
    
    end_time = time.time()
    print(f"[INFO] Relation extraction completed in {end_time - start_time:.2f} seconds.")
    print(f"[INFO] Extracted {len(final_triples)} final relations.")
    
    return final_triples

# --- Utility Functions (Optional) ---
def generate_graphviz_graph(text: str, output_path: str = "graph.png", use_enhanced_settings: bool = True):
    """
    Generates a Graphviz visualization of OpenIE triples from text.
    Note: This function from the 'openie' library directly calls annotate.
    It might not use the caching or specific client instances managed here.
    For consistency, it might be better to extract triples first using
    extract_relations_from_paragraph and then build a graph with another library.
    However, providing the original functionality for convenience.
    """
    print(f"[INFO] Attempting to generate Graphviz graph for text (output: {output_path})...")
    client = get_stanford_oie_client(use_enhanced_settings) # Gets a client
    if not client:
        print("[ERROR] Cannot generate graph: OpenIE client not available.")
        return False
    
    # The openie-python wrapper's generate_graphviz_graph itself calls annotate.
    # It doesn't take pre-extracted triples.
    try:
        # We need to pass the text directly, not pre-extracted triples.
        # The properties used will be those the client was initialized with.
        client.generate_graphviz_graph(text, output_path)
        if os.path.exists(output_path):
            print(f"[INFO] Graphviz graph generated successfully at '{output_path}'.")
            return True
        else:
            print(f"[WARNING] Graphviz graph generation command executed, but output file not found at '{output_path}'. Check Graphviz installation and PATH.")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to generate Graphviz graph: {e}")
        print(traceback.format_exc())
        return False

# --- Example Usage ---
if __name__ == "__main__":
    _ensure_corenlp_home() # Ensure CORENLP_HOME is set for the test run

    print("-" * 50)
    print("Example 1: Short sentence (Enhanced Settings)")
    sentence1 = "The quick brown fox jumps over the lazy dog."
    relations1 = extract_relations_from_paragraph(sentence1, use_enhanced_settings=True)
    for rel in relations1:
        print(f"  S: {rel['subject']}, R: {rel['relation']}, O: {rel['object']}")

    print("-" * 50)
    print("Example 2: Slightly more complex sentence (Standard Settings)")
    sentence2 = "Stanford University is located in California and is a renowned research institution."
    relations2 = extract_relations_from_paragraph(sentence2, use_enhanced_settings=False)
    for rel in relations2:
        print(f"  S: {rel['subject']}, R: {rel['relation']}, O: {rel['object']}")

    print("-" * 50)
    print("Example 3: Short paragraph (8-10 sentences style) - Enhanced Settings")
    paragraph = (
        "Climate change is a significant global issue. "
        "It is primarily caused by human activities, such as burning fossil fuels and deforestation. "
        "These activities release greenhouse gases into the atmosphere. "
        "The increased concentration of these gases traps heat, leading to a gradual warming of the Earth's climate. "
        "Consequences include rising sea levels, more frequent extreme weather events, and disruptions to ecosystems. "
        "International agreements, like the Paris Agreement, aim to mitigate these effects. "
        "Countries are encouraged to reduce their carbon emissions and invest in renewable energy sources. "
        "Individual actions also play a role in addressing this complex challenge."
    )
    relations_paragraph = extract_relations_from_paragraph(paragraph)
    print(f"Found {len(relations_paragraph)} relations in the paragraph:")
    for i, rel in enumerate(relations_paragraph):
        print(f"  {i+1}. S: {rel['subject']}, R: {rel['relation']}, O: {rel['object']}")
    
    print("-" * 50)
    print("Example 4: Testing graph generation (if Graphviz is installed and in PATH)")
    graph_text = "Barack Obama was born in Hawaii. He served as the 44th U.S. President."
    # generate_graphviz_graph(graph_text, "obama_graph.png") # Uncomment to test

    # Ensure clients are closed if script is long-running and not exiting immediately
    # atexit handles this for normal script termination.
    # For interactive sessions, call close_oie_clients() manually if needed.
    # close_oie_clients() # Usually not needed here due to atexit

    print("-" * 50)
    print("[INFO] OIE script execution finished.")
    print("[INFO] Remember to call close_oie_clients() if using in a long-running application where atexit might not cover all exit paths.")
    passage = "Last year, commercial companies, military and civil departments and amateurs sent more than 400 satellites into orbit, over four times the yearly average in the previous decade. Numbers could rise even more sharply if leading space companies follow through on plans to deploy hundreds to thousands of large constellations of satellites to space in the next few years."
    relations_paragraph = extract_relations_from_paragraph(passage)
    
    print(f"Found {len(relations_paragraph)} relations in the paragraph:")
    for i, rel in enumerate(relations_paragraph):
        print(f"  {i+1}. S: {rel['subject']}, R: {rel['relation']}, O: {rel['object']}")
