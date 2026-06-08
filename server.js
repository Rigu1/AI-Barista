require('dotenv').config();

const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises; 
const { GoogleGenerativeAI } = require('@google/generative-ai');

const app = express();
const upload = multer({ dest: 'uploads/' });

app.use(express.json());
app.use(express.static('public'));

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

const runPython = (filePath, startTime, duration) => {
    return new Promise((resolve, reject) => {
        const scriptPath = path.join(__dirname, 'analyzer.py');
        const pythonProcess = spawn('python3', [scriptPath, filePath, startTime, duration]);

        let dataString = '';
        let errorString = '';

        pythonProcess.stdout.on('data', (data) => { dataString += data.toString(); });
        pythonProcess.stderr.on('data', (data) => { errorString += data.toString(); });

        pythonProcess.on('close', (code) => {
            if (code !== 0 || errorString) {
                console.error(`[Python Error]:\n${errorString}`);
            }
            try {
                const jsonStart = dataString.indexOf('{');
                const jsonEnd = dataString.lastIndexOf('}') + 1;
                
                if (jsonStart === -1 || jsonEnd === 0) {
                    throw new Error("Python script did not return valid JSON.");
                }
                resolve(JSON.parse(dataString.substring(jsonStart, jsonEnd)));
            } catch (e) {
                reject(new Error(errorString || "JSON parsing failed."));
            }
        });
    });
};

app.post('/api/analyze', upload.single('audio'), async (req, res) => {
    const { start_time, duration } = req.body;
    const filePath = req.file.path;

    console.log(`[Server] Request received. Analyzing audio from ${start_time}s for ${duration}s.`);

    try {
        const result = await runPython(filePath, start_time, duration);
        console.log('[Server] Analysis complete. Returning payload to client.');
        res.json(result);
    } catch (error) {
        console.error("[Node.js Error]:", error.message);
        res.status(500).json({ error: "Internal server error during audio analysis." });
    } finally {
        try {
            await fs.unlink(filePath);
            console.log('[Server] Temporary file cleaned up successfully.');
        } catch (e) {
            console.error("[Server] Failed to clean up temporary file:", e);
        }
    }
});

app.post('/api/recommend', async (req, res) => {
    const { genres } = req.body;
    
    console.log(`[Server] Request received. Requesting LLM recommendation based on genres.`);

    try {
        const model = genAI.getGenerativeModel({ 
            model: "gemini-3.5-flash"
        });
        
        const prompt = `당신은 'Café de Music'의 친절하고 감성적인 AI 바리스타입니다. 손님이 다음 음악 장르 비율(테이스팅 노트)을 가진 음악을 들려주었습니다.
장르 데이터: ${JSON.stringify(genres)}

이 취향을 가진 손님에게 어울리는 실제 곡 3개를 추천해주세요. 따뜻한 카페 바리스타의 말투로, 곡마다 추천 이유를 짧고 예쁘게 작성해주세요. 너무 길지 않게 핵심만 다정하게 말해주세요.`;

        const result = await model.generateContent(prompt);
        const response = await result.response;
        const text = response.text();

        console.log('[Server] LLM response generated successfully.');
        res.json({ recommendation: text });
    } catch (error) {
        console.error("[LLM Error]:", error);
        res.status(500).json({ error: "Failed to generate LLM recommendation." });
    }
});

app.listen(3000, () => {
    console.log('Server is running on http://localhost:3000');
});