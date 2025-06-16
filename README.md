Ever found yourself watching a YouTube video, maybe a lecture, tech tutorial, or product review and had a specific question pop into your head?

🔍 "Wait, what did they say about transformer models again?"
🔍 "When did they mention the architecture name?"

Instead of scrubbing through the whole video, I built a solution:

 💬 A YouTube Chatbot that answers your questions directly from the video transcript!

💡 Key Features:
 ✅ Accepts any YouTube URL with captions
 ✅ Extracts and processes the transcript (even with auto-generated captions!)
 ✅ Splits the text into smart chunks for accurate retrieval
 ✅ Embeds the data using OpenAI and stores it in a FAISS vector database
 ✅ Uses LangChain and GPT-3.5 to answer your questions contextually

If the video doesn't have captions enabled, no problem! The bot uses yt-dlp to fetch auto-generated subtitles and converts them for analysis. 🛠️

This was an exciting challenge, mixing NLP, vector databases, and real-world user interaction via Streamlit.

Want to try it as well? DM me, and I’d love to share the demo or walk through the code!
