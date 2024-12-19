from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

def search_youtube_videos(query, max_results=30):
    """Busca videos en YouTube y devuelve sus enlaces."""
    api_key = "AIzaSyDkFTX8oc9lfKR6zzUCspYu2mEKDXjlxc0" 
    youtube = build("youtube", "v3", developerKey=api_key)

    search_response = youtube.search().list(
        q=query,
        part="id,snippet",
        maxResults=max_results
    ).execute()

    video_links = []
    for item in search_response.get("items", []):
        if item["id"]["kind"] == "youtube#video":
            video_id = item["id"]["videoId"]
            video_links.append(f"https://www.youtube.com/watch?v={video_id}")

    return video_links

def save_transcription(video_url, output_file):
    """Obtiene los subtítulos de un video y los guarda en un archivo .txt."""
    try:
        # Extraer el ID del video de la URL
        video_id = video_url.split("v=")[1]

        # Obtener los subtítulos automáticos
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["es", "en"])

        # Formatear los subtítulos como texto plano
        formatter = TextFormatter()
        text = formatter.format_transcript(transcript)

        # Guardar los subtítulos en un archivo .txt
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(text)

        print(f"Transcripción guardada en: {output_file}")

    except Exception as e:
        print(f"Error al procesar {video_url}: {e}")

if __name__ == "__main__":
    query = "Reforma al Poder Judicial "
    video_links = search_youtube_videos(query, max_results=5)

    print("Enlaces encontrados:")
    for index, link in enumerate(video_links):
        print(link)
        output_file = f"transcription_Ref{index + 1}.txt"
        save_transcription(link, output_file)