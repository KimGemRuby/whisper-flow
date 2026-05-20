""" transcriber """

import os
import asyncio

import torch
import numpy as np

import whisper
from whisper import Whisper


models = {}


def get_model(model_name="base") -> Whisper:
    """load whisper model by name (tiny, base, small, medium, large, turbo)"""
    if model_name not in models:
        models[model_name] = whisper.load_model(model_name).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return models[model_name]


def transcribe_pcm_chunks(
    model: Whisper, chunks: list, lang="en", temperature=0.1, log_prob=-0.5
) -> dict:
    """transcribes pcm chunks list"""
    arr = (
        np.frombuffer(b"".join(chunks), np.int16).flatten().astype(np.float32) / 32768.0
    )
    return model.transcribe(
        arr,
        fp16=False,
        language=lang,
        logprob_threshold=log_prob,
        temperature=temperature,
    )


async def transcribe_pcm_chunks_async(
    model: Whisper, chunks: list, lang="en", temperature=0.1, log_prob=-0.5
) -> dict:
    """transcribes pcm chunks async"""
    return await asyncio.get_running_loop().run_in_executor(
        None, transcribe_pcm_chunks, model, chunks, lang, temperature, log_prob
    )
