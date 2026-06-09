require('dotenv').config();

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs').promises; 
const { GoogleGenerativeAI } = require('@google/generative-ai');

const app = express();
const upload = multer({ dest: 'uploads/' });

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));
app.use(express.static('public'));

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

app.post('/api/analyze', upload.single('audio'), async (req, res) => {
    const { start_time, duration } = req.body;
    const filePath = path.resolve(req.file.path)

    console.log(`[Node.js] 오디오 파일 도착, 분석을 요청합니다.`);

    try {
        const pyResponse = await fetch('http://127.0.0.1:8000/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filepath: filePath,
                start_time: parseFloat(start_time),
                duration: parseFloat(duration)
            })
        });

        if (!pyResponse.ok) {
            const errorData = await pyResponse.text();
            throw new Error(`FastAPI에서 에러 발생: ${errorData}`);
        }
        
        const result = await pyResponse.json();
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
    const { genres } = req.body;
    
    console.log(`[Node.js] 테이스팅 노트 도착! Gemini 바리스타에게 추천 곡을 묻습니다.`);

    try {
        const model = genAI.getGenerativeModel({ model: "gemini-3.5-flash" });
        
        const prompt = `당신은 'Café de Music'의 친절하고 감성적인 AI 바리스타입니다. 손님이 다음 음악 장르 비율(테이스팅 노트)을 가진 음악을 들려주었습니다.
장르 데이터: ${JSON.stringify(genres)}

이 취향을 가진 손님에게 어울리는 실제 곡 3개를 추천해주세요. 따뜻한 카페 바리스타의 말투로, 곡마다 추천 이유를 짧고 예쁘게 작성해주세요. 너무 길지 않게 핵심만 다정하게 말해주세요.`;

        const result = await model.generateContent(prompt);
        const response = await result.response;
        const text = response.text();

        console.log('[Node.js] AI 바리스타 추천 완료!');
        res.json({ recommendation: text });
    } catch (error) {
        console.error("[Gemini 에러]:", error);
        res.status(500).json({ error: "AI 바리스타의 큐레이션 생성에 실패했습니다." });
    }
});

app.listen(3000, () => {
    console.log('✨ [Node.js] 메인 서버 기동');
});