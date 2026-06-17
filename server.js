require('dotenv').config();

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs').promises; 
const { GoogleGenerativeAI } = require('@google/generative-ai');
const http = require('http');
const https = require('https');

const app = express();
const upload = multer({ dest: 'uploads/' });

// 연결 재사용(Keep-Alive) 에이전트 설정
const httpAgent = new http.Agent({ keepAlive: true, keepAliveMsecs: 1000 });
const httpsAgent = new https.Agent({ keepAlive: true, keepAliveMsecs: 1000 });

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));
app.use(express.static('public'));

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

function parseJsonResponse(text) {
    const trimmed = text.trim();
    const withoutFence = trimmed
        .replace(/^```(?:json)?\s*/i, '')
        .replace(/\s*```$/i, '');

    return JSON.parse(withoutFence);
}

function formatRecommendations(recommendations) {
    return recommendations.slice(0, 3).map((item, index) => {
        const title = escapeHtml(item.title);
        const artist = escapeHtml(item.artist);
        const reason = escapeHtml(item.reason);
        const query = encodeURIComponent(`${item.title ?? ''} ${item.artist ?? ''}`.trim());
        const youtubeUrl = `https://www.youtube.com/results?search_query=${query}`;

        return `${index + 1}. <strong>${title} - ${artist}</strong> <a class="youtube-link" href="${youtubeUrl}" target="_blank" rel="noopener noreferrer">YouTube</a><br>${reason}`;
    }).join('<br><br>');
}

app.post('/api/analyze', upload.single('audio'), async (req, res) => {
    const { start_time, duration } = req.body;
    const filePath = path.resolve(req.file.path);

    console.log(`[Node.js] 오디오 파일 도착, 분석을 요청합니다.`);

    try {
        console.time('FastAPI-Latency'); 
        
        const pyResponse = await fetch('http://127.0.0.1:8000/analyze', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Connection': 'keep-alive' 
            },
            agent: httpAgent,
            body: JSON.stringify({
                filepath: String(filePath),
                start_time: parseFloat(start_time),
                duration: parseFloat(duration)
            })
        });

        if (!pyResponse.ok) {
            const errorData = await pyResponse.text();
            throw new Error(`FastAPI에서 에러 발생: ${errorData}`);
        }
        
        const result = await pyResponse.json();
        console.timeEnd('FastAPI-Latency'); 
        
        console.log('[Node.js] 분석 완료! 손님에게 테이스팅 노트를 전달합니다.');
        res.json(result);

    } catch (error) {
        console.error("[Node.js 통신 에러]:", error.message);
        res.status(500).json({ error: "오디오 분석 중 내부 통신 에러가 발생했습니다." });
    } finally {
        try {
            await fs.unlink(filePath);
            console.log('[Node.js] 다 쓴 임시 오디오 파일을 깨끗하게 청소했습니다.');
        } catch (e) {
            console.error("[Node.js] 파일 청소 실패:", e);
        }
    }
});

app.post('/api/recommend', async (req, res) => {
    const { genres, trackName } = req.body;
    const requestId = Date.now();
    const timerLabel = `Gemini-Latency-${requestId}`;
    
    console.log(`[Node.js] 테이스팅 노트 도착! Gemini 바리스타에게 추천 곡을 묻습니다.`);

    try {
        console.time(timerLabel);
        const model = genAI.getGenerativeModel({
            model: "gemini-3.1-flash-lite",
            generationConfig: {
                responseMimeType: "application/json"
            }
        });
        
        const prompt = `당신은 'Café de Music'의 친절하고 감성적인 AI 바리스타입니다. 손님이 다음 음악 장르 비율(테이스팅 노트)을 가진 음악을 들려주었습니다.
업로드한 파일명: ${trackName || '알 수 없음'}
장르 데이터: ${JSON.stringify(genres)}

이 취향을 가진 손님에게 어울리는 실제 곡 3개를 추천해주세요.
업로드한 파일명은 아주 약한 힌트로만 참고하고, 추천의 주된 근거는 장르 데이터로 삼아주세요.
파일명만 보고 원곡, 아티스트, 분위기를 단정하지 마세요.
파일명이 audio.mp3, track01.wav, recording, KakaoTalk, 숫자/날짜 위주 이름처럼 의미 없는 파일명으로 보이면 완전히 무시해주세요.
추천 곡은 실제로 존재한다고 확실히 알고 있는 곡만 골라주세요.
확신이 없으면 파일명 추론을 포기하고, 장르 데이터에 어울리는 널리 알려진 실제 곡을 추천해주세요.
반드시 아래 JSON 형식만 반환해주세요. 마크다운, 설명, 코드블록은 넣지 마세요.
{
  "recommendations": [
    { "title": "곡명", "artist": "아티스트명", "reason": "따뜻한 카페 바리스타 말투의 짧은 추천 이유" }
  ]
}`;

        const result = await model.generateContent(prompt);
        const response = await result.response;
        const text = response.text();
        const parsed = parseJsonResponse(text);
        const formattedRecommendation = formatRecommendations(parsed.recommendations || []);

        console.timeEnd(timerLabel);
        console.log('[Node.js] AI 바리스타 추천 완료!');
        
        res.json({ recommendation: formattedRecommendation });
    } catch (error) {
        console.error("[Gemini 에러]:", error);
        res.status(500).json({ error: "AI 바리스타의 큐레이션 생성에 실패했습니다." });
    }
});

const server = app.listen(3000, '127.0.0.1', () => {
    console.log('✨ [Node.js] 메인 서버 기동');
});
