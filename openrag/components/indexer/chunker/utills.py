from langchain_core.documents.base import Document
from langchain_openai import ChatOpenAI
import re


# Regex to match a Markdown table (header + delimiter + at least one row)
TABLE_RE = re.compile(
    r"(\n\|.*?\|\r?\n\|\s*[:-]+(?:\s*\|[:-]+)*\|\r?\n(?:\|.*?\|\r?\n)+)", re.DOTALL
)

# Regex to match image descriptions
IMAGE_RE = re.compile(r"(<image_description>(.*?)</image_description>)", re.DOTALL)


def _get_token_length(documents: str | Document, llm: ChatOpenAI = None) -> int:
    """Calculate the total number of tokens in a list of documents."""
    length_func = llm.get_num_tokens if llm else lambda x: len(x)
    num_tokens = []
    for doc in documents:
        if isinstance(doc, str):
            num_tokens.append(length_func(doc))
        elif isinstance(doc, Document):
            num_tokens.append(length_func(doc.page_content))
        else:
            raise ValueError("Documents must be either str or Document instances.")
    return sum(num_tokens)


def combine_chunks(
    chunks: list[str | Document],
    llm: ChatOpenAI = None,
    chunk_max_size: int = 512,
) -> list[str]:  # type: ignore
    doc_n_tokens = map(lambda doc: _get_token_length(documents=[doc], llm=llm), chunks)

    # regroup subsequent chunks based if the total length is less than token_max
    grouped_docs = []
    current_group = []
    current_length = 0

    for doc, n_tokens in zip(chunks, doc_n_tokens):
        if isinstance(doc, Document):
            doc = doc.page_content

        if current_length + n_tokens > chunk_max_size:
            if current_group:
                grouped_docs.append(current_group)
            current_group = [doc]
            current_length = n_tokens
        else:
            current_group.append(doc)
            current_length += n_tokens

    if current_group:
        grouped_docs.append(current_group)

    return ["\n".join(group) for group in grouped_docs]


def span_inside(span, container):
    return container[0] <= span[0] and span[1] <= container[1]


def split_md_elements(md_text: str):
    """
    Split markdown text into segments of text, tables, and images.
    Returns a list of tuples: (type, content) where type is 'text', 'table', or 'image'
    """
    all_matches = []

    # Find image matches first and record their spans
    image_spans = []
    for match in IMAGE_RE.finditer(md_text):
        span = match.span()
        all_matches.append((span, "image", match.group(1)))
        image_spans.append(span)

    # Find table matches, but skip those that are fully inside an image description
    for match in TABLE_RE.finditer(md_text):
        span = match.span()
        if not any(span_inside(span, image_span) for image_span in image_spans):
            all_matches.append((span, "table", match.group(1)))

    # Sort matches by start position
    all_matches.sort(key=lambda x: x[0][0])

    parts = []
    last = 0

    for (start, end), match_type, content in all_matches:
        # Add text segment before this match if there is any
        if start > last:
            text_segment = md_text[last:start]
            if text_segment.strip():  # Only add non-empty text segments
                parts.append(("text", text_segment.strip()))

        # Add the matched segment
        parts.append((match_type, content.strip()))
        last = end

    # Add remaining text after the last match
    if last < len(md_text):
        remaining_text = md_text[last:]
        if remaining_text.strip():  # Only add non-empty text segments
            parts.append(("text", remaining_text.strip()))

    return parts
