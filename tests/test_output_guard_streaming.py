from src.llm.output_guard import OutputGuard
from src.llm.sentence_buffer import SentenceBuffer


def test_streaming_guard_preserves_spaces_between_token_chunks():
    guard = OutputGuard()
    buffer = SentenceBuffer()

    emitted = []
    for chunk in ["I'm ", "here ", "to help ", "you out."]:
        clean = guard.process_chunk(chunk)
        emitted.append(clean)
        buffer.feed(clean)
    emitted.append(guard.flush())

    assert "".join(emitted) == "I'm here to help you out."