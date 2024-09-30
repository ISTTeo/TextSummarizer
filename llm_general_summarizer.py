import json
import os
from utility.phi3 import model_tokenizer, run_phi3, get_token_count
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize

def chunk_text(tokenizer, text, max_tokens=3000):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence)
        sentence_token_count = len(sentence_tokens)
        
        if current_token_count + sentence_token_count > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_token_count = sentence_token_count
        else:
            current_chunk.append(sentence)
            current_token_count += sentence_token_count
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
    
def process_transcript_in_chunks(tokenizer, artifact_original_text, max_tokens_per_part=3000):
    max_model_length = tokenizer.model_max_length
    max_tokens_per_part = min(max_tokens_per_part, max_model_length - 2)  # -2 for special tokens

    def chunk_generator(text, chunk_size=10000):
        """Yield chunks of text"""
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]

    grouped_segments = []
    current_group = []
    current_token_count = 0

    for chunk in chunk_generator(artifact_original_text):
        sentences = sent_tokenize(chunk)
        
        for sentence in sentences:
            sentence_token_count = len(tokenizer.tokenize(sentence))
            
            if current_token_count + sentence_token_count > max_tokens_per_part:
                if current_group:
                    grouped_segments.append(' '.join(current_group))
                    yield ' '.join(current_group)  # Yield the group for processing
                current_group = []
                current_token_count = 0
            
            current_group.append(sentence)
            current_token_count += sentence_token_count
    
    # Yield the last group if it exists
    if current_group:
        yield ' '.join(current_group)

def summarize_with_context(segment, context, model, tokenizer, generation_args, sys_summarize_with_context):
    sys = f"""
    This is a segment of an bigger textual artifact. The previous context is:
    {context}
    Instructions:\n
    """
    sys += sys_summarize_with_context
    user = f"{segment}"
    string = run_phi3(sys, user, model, tokenizer, generation_args)[-1]['content']
    
    return string

def extract_insights_with_context(text, context, model, tokenizer, generation_args, sys_command_extract_with_context):
    sys = f"""
    This is a segment of an bigger textual artifact. The previous context is:
    {context}
    Instructions:\n
    """
    sys += sys_command_extract_with_context
    user = f"{text}"
    insights = run_phi3(sys, user, model, tokenizer, generation_args)[-1]['content']
    return insights

def get_first_summary(m, t, generation_args, artifact, original_artifact_text_field, sys_summarize_with_context, any_general_context=""):
    artifact_original_text = artifact[original_artifact_text_field]
    segment_generator = process_transcript_in_chunks(t, artifact_original_text, max_tokens_per_part=3000)

    summaries = []
    context = f"This is the start of the artifact summary. "
    context += f"This is some general context for this artifact:\n {artifact[any_general_context]}" if artifact[any_general_context] else "There's no context at first"

    for segment in segment_generator:
        summary = summarize_with_context(segment, context, m, t, generation_args, sys_summarize_with_context)
        summaries.append(summary)
        
        # Update context for next iteration
        # We'll use the last 2-3 sentences of the current summary as context
        context_sentences = sent_tokenize(summary)[-3:]  # Get last 3 sentences
        context = " ".join(context_sentences)

    return "\n".join(summaries)

def get_summary_over_summary(m, t, generation_args, artifact, original_summary_field, sys_command_extract_with_context, any_general_context=""):
    previous_summary = artifact[original_summary_field]
    chunks = chunk_text(t, previous_summary)

    insights_list = []
    context = f"This is the start of the artifact summary."
    context += f"This is some general context for this artifact:\n {artifact[any_general_context]}" if artifact[any_general_context] else "There's no context at first"

    for chunk in chunks:
        insights = extract_insights_with_context(chunk, context, m, t, generation_args, sys_command_extract_with_context)
        insights_list.append(insights)
        
        # Update context for next iteration
        context_sentences = sent_tokenize(insights)[-3:]  # Get last 3 sentences
        context = " ".join(context_sentences)

    # Combine all insights
    all_insights = "\n\n".join(insights_list)

    # Final summarization of insights with overall context
    return extract_insights_with_context(all_insights, context, m, t, generation_args)

