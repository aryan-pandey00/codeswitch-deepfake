# Web frontend

Upload or record a clip, get two verdicts on it. The interface exists to make one comparison
visible: a detector trained on a single corpus against the same detector trained with a little
recording variety, scoring the same audio.

## Running locally

```
npm install
cp .env.example .env
npm run dev
```

`VITE_API_URL` points at the deployed API. Leaving it empty makes requests relative, which is
only useful behind a proxy.

## Deploying

Vercel, framework preset Vite, root directory `app/web`. Set `VITE_API_URL` as an environment
variable in the project settings. Vite inlines it at build time, so changing it needs a
redeploy rather than a restart.

## Behaviour worth knowing

The API is contacted once on load. That doubles as a warm-up, so the first real request is not
the one paying for a cold container.

If the API cannot be reached, the app renders a worked example instead of an error. A
portfolio link that shows what the project does still demonstrates it; a dead link does not.
The status line says plainly which of the two is happening, so the example is never mistaken
for a live result.

Recording produces WebM in most browsers. The API decodes it with ffmpeg rather than expecting
WAV.
