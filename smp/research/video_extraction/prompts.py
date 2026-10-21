"""Extraction designs compared in this study.

Each variant is a prompt plus an output schema. All variants share RULES, so they
differ only in what they ask for and how the output is organised. The text they
produce is meant to replace the video in later analysis, so the comparison asks
how much of the video survives in the text, and how accurately.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

RULES = """\
Rules:
- Describe only what is actually seen or heard. If something is unclear, write
  "unclear" rather than guessing.
- Be neutral and factual. Do not judge quality, creativity or appeal, and do not
  say or imply how popular, trending, viral, famous or well known the video, its
  sound, its format or anyone in it is.
- Do not infer race, ethnicity, nationality, religion, sexual orientation or
  health. Describe people by visible, neutral attributes only: apparent age band
  (child, teen, adult, older adult), clothing, hair, actions.
- Do not name real people unless the name is shown on screen or spoken.
- Do not transcribe usernames, handles, watermarks, or view, like or follower
  numbers shown on screen.
- Quote speech and on-screen text in the original wording.
- Times are seconds from the start of the video."""

CONTENT_TYPES = (
    "tutorial or how-to; recipe or food preparation; product review or demo; "
    "unboxing or haul; comedy skit; prank; lip-sync or dance to audio; music "
    "performance; vlog or day in the life; story time or talking head; reaction; "
    "educational or facts; beauty, fashion or styling; fitness; pets or animals; "
    "sports; gaming; art, craft or DIY; travel or scenery; ASMR; slideshow or "
    "photo montage; other"
)


# --- V1: field schema (the current pipeline's design, with the rule fixes) ---

V1_PROMPT = f"""\
You are annotating short-form social media videos for a research dataset.
Watch and listen to the whole video, then describe it in the JSON schema provided.

{RULES}
- Keep each field short; fill every field."""


class V1Fields(BaseModel):
    summary: str = Field(description="2-3 sentences: what happens, start to finish.")
    content_type: str = Field(description=f"The single best fit from: {CONTENT_TYPES}.")
    topic: str = Field(description="The subject in a few words.")
    hook: str = Field(description="What is seen and heard in the first 3 seconds.")
    structure: str = Field(description="How it unfolds, e.g. setup then punchline, step-by-step list, before/after reveal, single continuous shot, montage.")
    key_moment: str = Field(description="The main payoff, twist, reveal or punchline and roughly when, or 'none'.")
    people: str = Field(description="How many people, neutral visible description, and whether anyone speaks to camera; 'none' if no people.")
    setting: str = Field(description="The specific place when visible, e.g. home kitchen, car interior, gym, beach.")
    filming_style: str = Field(description="e.g. handheld selfie, tripod, screen recording, professional camera, green screen, edited stock footage.")
    editing_pace: str = Field(description="slow / medium / fast, with a rough number of cuts.")
    on_screen_text: str = Field(description="The main text overlays, quoted, or 'none'.")
    speech: str = Field(description="Who speaks (on camera, voice-over, none), a summary, and one quote.")
    audio: str = Field(description="Background audio: speech only, song with vocals, instrumental, sound effects, ambient, silent.")
    tone: str = Field(description="Emotional tone, e.g. funny, wholesome, informative, dramatic, calm, sarcastic.")
    call_to_action: str = Field(description="Explicit requests to follow, like, comment, share, buy or visit a link, quoted, or 'none'.")
    products_or_brands: str = Field(description="Products, brands or businesses clearly shown or mentioned, or 'none'.")


# --- V2: one free-text narrative ---

V2_PROMPT = f"""\
Watch and listen to the whole video. Then write a detailed description for a
reader who will never see or hear it, so that they know everything that happens:
what is shown, who appears and what they do, what is said and written on screen,
the sounds and music, the setting, how it is filmed and edited, and the order in
which things happen.

{RULES}
- Write continuous prose, in the order things happen. Be as complete as the
  video requires; a short video needs fewer words."""


class V2Narrative(BaseModel):
    description: str = Field(description="The full description, as prose.")


# --- V3: timeline ---

V3_PROMPT = f"""\
Watch and listen to the whole video. Break it into consecutive segments (a new
segment at each cut, scene change, or clear change in action) and describe each
one. Then give a one-sentence summary.

{RULES}"""


class Segment(BaseModel):
    start: float = Field(description="Start time in seconds.")
    end: float = Field(description="End time in seconds.")
    visual: str = Field(description="What is shown: people, actions, objects, setting, camera.")
    speech: str = Field(description="Words spoken in this segment, verbatim, with who says them; 'none' if none.")
    on_screen_text: str = Field(description="Text shown in this segment, verbatim; 'none' if none.")
    audio: str = Field(description="Music and sounds in this segment; 'none' if silent.")


class V3Timeline(BaseModel):
    segments: List[Segment]
    summary: str = Field(description="One sentence: what the video is.")


# --- V4: full record (attributes + timeline + verbatim channels) ---

V4_PROMPT = f"""\
You are writing the complete text record of a short-form social media video.
The record will replace the video in a research dataset: later analysis will see
only this text, never the video. Capture everything a viewer would perceive.

Watch and listen to the whole video, then fill the JSON schema provided:
overall attributes, a timeline of consecutive segments, the full spoken
transcript, every piece of on-screen text, and the audio.

{RULES}
- The transcript and on-screen text must be complete and verbatim, not
  summarised. If speech is too unclear to transcribe, write "unclear" for that part."""


class Utterance(BaseModel):
    time: float = Field(description="Approximate start time in seconds.")
    speaker: str = Field(description="Who speaks, e.g. 'person on camera', 'voice-over', 'second person', 'audio track lip-synced'.")
    text: str = Field(description="The words, verbatim.")


class TextOverlay(BaseModel):
    time: float = Field(description="Approximate time it first appears, in seconds.")
    text: str = Field(description="The text, verbatim.")


class V4Record(BaseModel):
    summary: str = Field(description="2-3 sentences: what happens, start to finish.")
    content_type: str = Field(description=f"The single best fit from: {CONTENT_TYPES}.")
    topic: str = Field(description="The subject in a few words.")
    format: str = Field(description="How it is built, e.g. setup then punchline, step-by-step list, before/after reveal, single continuous shot, montage, POV.")
    hook: str = Field(description="Exactly what is seen and heard in the first 3 seconds.")
    people: str = Field(description="Each person who appears: neutral visible description, role, whether they speak to camera. 'none' if no people.")
    setting: str = Field(description="The specific place(s) when visible.")
    filming: str = Field(description="Camera and production: handheld selfie, tripod, screen recording, professional camera, green screen, stock footage; framing; lighting.")
    editing: str = Field(description="Number of cuts, pace, transitions, effects, filters, captions style.")
    segments: List[Segment] = Field(description="Consecutive segments covering the whole video.")
    transcript: List[Utterance] = Field(description="Every spoken line, verbatim, in order; empty list if no speech.")
    on_screen_text: List[TextOverlay] = Field(description="Every text overlay, caption or sign, verbatim, in order; empty list if none.")
    audio: str = Field(description="Music (vocals or instrumental, genre, tempo, mood), sound effects, ambient sound, silence; name a song only if its title is shown or sung clearly.")
    tone: str = Field(description="Emotional tone.")
    call_to_action: str = Field(description="Explicit requests to follow, like, comment, share, buy or visit a link, quoted, or 'none'.")
    products_or_brands: str = Field(description="Products, brands or businesses clearly shown or mentioned, or 'none'.")


VARIANTS = {
    # name: (prompt, schema, model, fps)
    "V1_fields":        (V1_PROMPT, V1Fields,    "gemini-3.5-flash", None),
    "V2_narrative":     (V2_PROMPT, V2Narrative, "gemini-3.5-flash", None),
    "V3_timeline":      (V3_PROMPT, V3Timeline,  "gemini-3.5-flash", None),
    "V4_record":        (V4_PROMPT, V4Record,    "gemini-3.5-flash", None),
    "V5_record_fps2":   (V4_PROMPT, V4Record,    "gemini-3.5-flash", 2.0),
    "V6_record_3.8":    (V4_PROMPT, V4Record,    "gemini-3.8-flash", None),
}


# --- Evaluation prompts (run by the judge model) ---

QUESTIONS_PROMPT = """\
You are building a test of whether a text description can stand in for a video.
Watch and listen to the whole video, then write 12 questions about it, each with
a short correct answer, that anyone who watched the video could answer and that
a good text description of the video should let a reader answer.

Cover these categories (use the counts shown; if the video has no speech or no
on-screen text, move those questions to other categories):
- visual (3): objects, actions, setting, appearance of what is shown
- people (2): who appears, what they do, how they present (no race, ethnicity,
  religion or names unless shown on screen)
- speech (2): what is said
- on_screen_text (2): what is written on screen
- audio (1): music or sounds
- sequence (2): order of events, timing, how it starts or ends, the payoff

Each question must be objective and specific to this video. Avoid questions
whose answer is a matter of opinion, and avoid anything about popularity.
Answers should be a few words."""


class QA(BaseModel):
    category: str = Field(description="visual, people, speech, on_screen_text, audio or sequence.")
    question: str
    answer: str


class QuestionSet(BaseModel):
    questions: List[QA]


ANSWER_PROMPT = """\
Below is a text record of a short video. You cannot see the video; use only the
text. Answer each question in a few words. If the text does not contain the
information, answer exactly "NOT IN TEXT". Do not guess.

TEXT RECORD:
{record}

QUESTIONS:
{questions}"""


class Answer(BaseModel):
    id: int
    answer: str


class AnswerSet(BaseModel):
    answers: List[Answer]


GRADE_PROMPT = """\
Grade each predicted answer against the reference answer for a question about a
video. Use:
- correct: same meaning as the reference (wording may differ)
- partial: partly right or less specific than the reference
- incorrect: contradicts the reference
- not_in_text: the prediction is "NOT IN TEXT"

{items}"""


class Grade(BaseModel):
    id: int
    grade: str = Field(description="correct, partial, incorrect or not_in_text")


class GradeSet(BaseModel):
    grades: List[Grade]


VERIFY_PROMPT = """\
Below is a text description that is supposed to describe this video accurately.
Watch and listen to the whole video, then check the description claim by claim.
List every claim that is contradicted by the video or that the video does not
support (including invented details, wrong quotes, wrong counts, wrong timing,
and anything stated with more certainty than the video allows). Ignore claims
that are correct. Mark each as major (changes what the video is about or what
happens) or minor (a small detail).

DESCRIPTION:
{record}"""


class Issue(BaseModel):
    claim: str
    problem: str
    severity: str = Field(description="major or minor")


class Verification(BaseModel):
    issues: List[Issue]
    accuracy: int = Field(description="Overall accuracy from 1 (mostly wrong) to 5 (fully accurate).")
