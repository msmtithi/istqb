import asyncio
import os
from pathlib import Path

import langdetect
from components.utils import SingletonMeta
from langchain_core.documents.base import Document
from openai import AsyncOpenAI
from pydub import AudioSegment, silence
from utils.logger import get_logger

from .base import BaseLoader

logger = get_logger()

MEDIA_FORMATS = [".wav", ".mp3", ".mp4", ".ogg", ".flv", ".wma", ".aac"]


class AudioTranscriber(metaclass=SingletonMeta):
    def __init__(self, config):
        self.client = AsyncOpenAI(
            base_url=config.loader.transcriber.base_url,
            api_key=config.loader.transcriber.api_key,
        )
        self.model_name = config.loader.transcriber.model_name
        self.max_chunk_ms = config.loader.transcriber.max_chunk_ms
        self.silence_thresh_db = config.loader.transcriber.silence_thresh_db
        self.min_silence_len_ms = config.loader.transcriber.min_silence_len_ms
        self.semaphore = asyncio.Semaphore(
            config.loader.transcriber.max_concurrent_chunks
        )

    async def transcribe(self, wav_path: Path) -> str:
        sound = AudioSegment.from_wav(wav_path)
        total_ms = len(sound)

        logger.info(f"Analyzing audio length {total_ms / 1000:.1f}s")

        # Split into chunks
        chunks = self._get_audio_chunks(sound)
        logger.info(f"Detected {len(chunks)} chunks")

        # Detect language
        language = await self._detect_language(
            sound[chunks[0][0] : chunks[0][1]], wav_path
        )
        logger.info(f"Detected language: {language}")

        # Create tasks for each chunk
        tasks = [
            self._process_chunk(i, sound[start:end], wav_path, language)
            for i, (start, end) in enumerate(chunks)
        ]

        texts = await asyncio.gather(*tasks)
        return self._stitch_transcriptions(texts)

    async def _process_chunk(
        self, index: int, segment: AudioSegment, wav_path: Path, language: str = None
    ) -> str:
        """Export a segment, transcribe it, and clean up."""
        async with self.semaphore:
            tmp_path = wav_path.parent / f"{wav_path.stem}_chunk_{index:03d}.wav"
            segment.export(tmp_path, format="wav")
            try:
                result = await self._transcribe_chunk(tmp_path, language)
                return result
            except Exception as e:
                logger.exception(
                    f"Error transcribing chunk {tmp_path.name}", error=str(e)
                )
                return ""
            finally:
                tmp_path.unlink(missing_ok=True)

    async def _transcribe_chunk(self, wav_path: Path, language: str = None) -> str:
        """Transcribe a single WAV chunk."""
        try:
            logger.debug(f"Transcribing chunk: {wav_path.name}")

            kwargs = {
                "model": self.model_name,
                "file": wav_path,
            }
            if language:
                kwargs["language"] = language

            result = await self.client.audio.transcriptions.create(**kwargs)
            return result.text.strip()
        except Exception as e:
            logger.exception(f"Error transcribing chunk {wav_path.name}", error=str(e))
            return ""

    def _get_audio_chunks(self, sound: AudioSegment) -> list[AudioSegment]:
        """Split audio into chunks based on silence detection."""
        total_ms = len(sound)
        if total_ms <= self.max_chunk_ms:
            return [sound]

        logger.debug("Detecting silences for chunking...")
        downsampled_sound = sound.set_channels(1).set_frame_rate(16000)
        silences = silence.detect_silence(
            downsampled_sound,
            min_silence_len=self.min_silence_len_ms,
            silence_thresh=self.silence_thresh_db,
        )
        silences = [(start, end) for start, end in silences]

        chunks = []
        start = 0
        while start < total_ms:
            target_end = start + self.max_chunk_ms
            if target_end >= total_ms:
                end = total_ms
            else:
                valid_silences = [s for s in silences if start < s[0] < target_end]
                if valid_silences:
                    end = valid_silences[-1][0]
                else:
                    end = target_end
            chunks.append((start, end))
            start = end

        return chunks

    def _stitch_transcriptions(self, texts: list[str]) -> str:
        """Concatenate with spacing."""
        return "\n".join(t.strip() for t in texts if t.strip())

    async def _detect_language(self, sound: AudioSegment, wav_path) -> str:
        """Detect the language of the audio segment."""
        tmp_path = wav_path.parent / f"{wav_path.stem}_langdetect.wav"
        sound.export(tmp_path, format="wav")
        text = await self._transcribe_chunk(tmp_path)
        try:
            return langdetect.detect(text)
        except Exception as e:
            logger.exception("Language detection failed", error=str(e))
            raise
        finally:
            os.remove(tmp_path)


class VideoAudioLoader(BaseLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transcriber = AudioTranscriber(config=self.config)

    async def aload_document(
        self, file_path, metadata: dict = None, save_markdown=False
    ):
        path = Path(file_path)
        if path.suffix not in MEDIA_FORMATS:
            logger.warning(
                f"This audio/video file ({path.suffix}) is not supported. "
                f"Supported formats: {MEDIA_FORMATS}"
            )
            return None

        # Convert to wav if needed
        if path.suffix == ".wav":
            audio_path_wav = path
        else:
            sound = AudioSegment.from_file(file=path, format=path.suffix[1:])
            audio_path_wav = path.with_suffix(".wav")
            sound.export(audio_path_wav, format="wav")

        content = await self.transcriber.transcribe(audio_path_wav)
        if path.suffix != ".wav":
            os.remove(audio_path_wav)

        doc = Document(page_content=content, metadata=metadata)
        if save_markdown:
            self.save_content(content, str(file_path))
        return doc
