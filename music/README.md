# Bach Chaconne BWV 1004 — source asset

The complete fifth movement, **Chaconne**, from Bach's second partita for
**solo violin** is supplied as an actual Standard MIDI File:
`bach-chaconne-bwv1004.mid`.

The source is [Mutopia piece 1426](https://www.mutopiaproject.org/cgibin/piece-info.cgi?id=1426),
Hajo Dezelski's 2019 LilyPond transcription of the Bach-Gesellschaft 1879 edition.
Attribution and license terms are recorded below.

## Original download URLs

- MIDI: <https://www.mutopiaproject.org/ftp/BachJS/BWV1004/bwv-1004_5/bwv-1004_5.mid>
- LilyPond: <https://www.mutopiaproject.org/ftp/BachJS/BWV1004/bwv-1004_5/bwv-1004_5.ly>

The supplied MIDI is unmodified. The original LilyPond header explicitly
identifies `5. Chaconne` and `Violine`; the score contains one violin staff with
four notated voices, not an accompaniment or orchestral arrangement.

## Verified MIDI facts

| Property | Value |
|---|---:|
| SMF format | 1 |
| Tracks | 2: one metadata track, one note track |
| Note-bearing MIDI channels | 1 (zero-based channel 0) |
| Note events | 3,085 complete note-on/note-off pairs |
| Native tempo | Quarter note = 60 BPM throughout |
| Time signature | 3/4 |
| Duration | 770 seconds (12:50) |
| Ticks per quarter note | 384 |
| Pitch range | MIDI 55–91 (G3–G6 with middle C = C4) |
| Maximum simultaneous held notes | 4 |
| Sustain pedal / pitch bends | None |

The file has **no program-change event**. Its score is for violin, but a generic
General MIDI player may use its default piano patch. Playback code should
explicitly select a violin voice (General MIDI program 40 zero-based, 41
one-based) or its own violin synthesis. Preserve the simultaneous notes and
overlaps; solo violin does not mean a monophonic sequence. This is a quantized
score rendering, not a recording of a human performance.

## Integrity and reproducibility

- MIDI SHA-256: `f4946c53a3658f6c436bb84976740deef6b60833db45d5e16a2608a77f1e7b70`
- LilyPond SHA-256: `c553876e6a194d575c8d1c2b045dc12b0e1a3b111a70bfa5a7d3f03cf11d941d`

Polyphony is measured using half-open note intervals: note-offs at a tick are
applied before note-ons at that same tick. Every note-on has a matching note-off.
Timing follows the original MIDI tempo map without tempo scaling.

## Attribution and license

**Johann Sebastian Bach — Partita II, BWV 1004, movement 5: Chaconne**,
for solo violin. Typeset by **Hajo Dezelski, © 2019**, for the **Mutopia Project**.
Mutopia publication identifier: **Mutopia-2019/05/30-1426**.
Edition used by the typesetter: Bach-Gesellschaft Edition, 1879, Band 27.1.

The MIDI transcription is distributed under
**Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)**.
The underlying Bach composition is public domain; the Mutopia transcription
retains its stated license and credit.

- Source and attribution: <https://www.mutopiaproject.org/cgibin/piece-info.cgi?id=1426>
- License: <https://creativecommons.org/licenses/by-sa/3.0/>
- Legal text: <https://creativecommons.org/licenses/by-sa/3.0/legalcode.en>
- Mutopia licensing information: <https://www.mutopiaproject.org/legal.html>

`bach-chaconne-bwv1004.mid` is an unchanged copy of the official
`bwv-1004_5.mid`, renamed locally. No notes, tempo events, channels or durations
were edited. The LilyPond source is linked above for provenance and is not
required to play or parse the MIDI.

When redistributing this asset, retain this credit and the license link.
Identify adaptations and distribute them under the applicable ShareAlike terms.
Do not imply endorsement by the composer, typesetter or Mutopia Project.

Mutopia provides its material as-is, without warranty, including any assurance
of musical or editorial accuracy. This notice preserves that source disclaimer.
It covers the music asset; it does not assign a license to unrelated application
code.
