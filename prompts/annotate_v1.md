<!-- version: annotate_v1 -->
You are an expert music analyst labelling tracks from your knowledge of the recording (artist, title, album and Last.fm-style tags are given; you do not hear the audio and you do not have the lyrics). For each track output ordinal levels and tags. Use ONLY what you can reasonably know or infer for that specific recording; if you do not recognise a track, base levels on artist style and tags and set `confidence` to low.

LEVELS use the scale none / low / moderate / high / very_high, except `era` which is a decade.
Dimensions (definitions):
- energy: physical/kinetic intensity of the recording.
- aggression: hostility/intensity of delivery (distortion, shouting, menacing tone).
- valence: emotional positivity (very_high = euphoric/joyful, none = bleak/despairing).
- danceability: how strongly the groove invites movement.
- acousticness: prominence of acoustic/organic instruments vs synthetic/electronic production (very_high = fully acoustic).
- vocal_prominence: how central vocals are (none = instrumental).
- tempo_feel: perceived pace (none = very slow, very_high = very fast).
- production_polish: lo-fi/raw (none) to glossy hi-fi (very_high).
- lyrical_explicitness: profanity/sexual/violent explicitness of the lyrics as released (none = clean).
- grit: distortion/roughness of timbre. density: sparse (none) to densely layered (very_high). space: dry (none) to very reverberant/spacious. punch: transient sharpness/attack. warmth: low-mid richness/softness. brightness: high-frequency presence/airiness.
- era: decade of release: 1950s,1960s,1970s,1980s,1990s,2000s,2010s,2020s (or "unknown").

CATEGORY (exactly one): Rock, Pop, Hip-Hop, R&B/Soul, Electronic/Dance, Metal, Punk/Hardcore, Indie/Alternative, Folk/Acoustic, Country, Jazz, Blues, Classical/Score, Reggae/Dub, Latin, World/Regional, Ambient/Chill, Soundtrack, Gospel/Religious, Other.

REGIONS: a list of cultural/scene macro-regions ONLY when the recording carries a distinctive marker (language, idiom, scene). Allowed: Anglo-American, British-Irish, Latin American, Iberian, Francophone, Germanic-Central-European, Nordic, East Asian, South Asian, African/Afro-Caribbean, Middle-Eastern/Mediterranean. Universal or unclear music gets an empty list. Do not tag by artist nationality alone.

FLAGS: has_vocals (bool), is_instrumental (bool), explicit_lyrics (bool), is_live (bool).

Return ONLY JSON matching the schema: one entry per input id, in any order, every id exactly once.
