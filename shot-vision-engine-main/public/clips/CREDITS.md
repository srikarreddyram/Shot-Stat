# Splash page footage

All clips are from [Mixkit](https://mixkit.co/free-stock-video/basketball/) and
are used under the **Mixkit License**: free for commercial and non-commercial
projects, no attribution required. The license does *not* permit redistributing
the clips as standalone files or using them in a competing stock service —
embedding them in this site is fine.

| File | Mixkit id | Shot |
| --- | --- | --- |
| 00-handle.mp4 | 33987 | Silhouetted player, smoke, backlit hoop |
| 01-contest.mp4 | 2273 | Low-angle drive at the rim |
| 02-shot.mp4 | 17206 | Rim from below, backlit |
| 03-drive.mp4 | 1216 | Overhead court — reads like a data view |
| 04-finish.mp4 | 751 | Overhead one-on-one, long shadows |
| 05-net.mp4 | 13733 | Rim and chain net, ball dropping through |

Selection rule: prefer silhouette, overhead and tight-detail shots. Wide
footage of recreational play reads as stock filler behind this layout, however
good the clip is on its own.

Posters (`*.jpg`) are single frames pulled from the matching clip.

These are generic players, not NBA footage. NBA game video is copyrighted by
the league and its broadcasters, and individual players' name and likeness
rights apply on top of that — neither can be used here.

## Swapping in your own clips

The splash page reads exactly six files from this folder, one per scroll
section. To use different footage, drop in an `.mp4` with the matching name —
no code change is needed:

| File | Section | Copy on that section |
| --- | --- | --- |
| `00-handle.mp4` | 0 | "Every bucket starts here." |
| `01-contest.mp4` | 1 | "The closeout is coming." |
| `02-shot.mp4` | 2 | "Not all shots are equal." |
| `03-drive.mp4` | 3 | "847,000 shots. 12 seasons." |
| `04-finish.mp4` | 4 | "XGBoost. Trained on mismatches." |
| `05-net.mp4` | 5 | "Know before you shoot." (CTA) |

Also replace the matching `.jpg`, which is the poster shown before the clip has
buffered. Any single frame at ~854px wide works.

Guidance: keep each file under ~3MB (720p is plenty behind a scrim), and prefer
clips that stay dark — the grade in `video-backdrop.tsx` darkens footage into
the palette, but it cannot rescue a blown-out daylight shot.

### Using NBA footage

NBA game video is owned by the league and its broadcasters, and individual
players hold name-and-likeness rights on top of that. Neither is covered by the
Mixkit clips here. Legitimate routes:

- **License it.** Getty Images carries NBA editorial video; the NBA also licenses
  footage directly. Licensed files drop straight into the table above.
- **Embed it.** Official NBA uploads on YouTube have embedding enabled, and an
  embedded player is a supported use. That has to be a real, visible player
  module though — YouTube's terms do not allow stripping the chrome and using
  it as a silent background layer, so it cannot replace this backdrop.

Do not pull clips from YouTube, NBA.com, or social media into this folder.
