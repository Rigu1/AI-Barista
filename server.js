require('dotenv').config();

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs').promises; 
const { GoogleGenerativeAI } = require('@google/generative-ai');
const http = require('http');
const https = require('https');

const app = express();
const MAX_UPLOAD_SIZE = 50 * 1024 * 1024;
const AUDIO_EXTENSIONS = new Set(['mp3', 'wav', 'm4a', 'aac', 'ogg', 'flac', 'webm']);
const PYTHON_ANALYZE_URL = 'http://127.0.0.1:8000/analyze';
const PYTHON_TIMEOUT_MS = 15000;
const GEMINI_TIMEOUT_MS = 20000;
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-3.1-flash-lite';

function isAcceptedAudioFile(file) {
    if (file.mimetype && file.mimetype.startsWith('audio/')) return true;

    const extension = path.extname(file.originalname || '').slice(1).toLowerCase();
    return AUDIO_EXTENSIONS.has(extension);
}

const upload = multer({
    dest: 'uploads/',
    limits: {
        fileSize: MAX_UPLOAD_SIZE,
        files: 1
    },
    fileFilter: (req, file, cb) => {
        if (isAcceptedAudioFile(file)) {
            cb(null, true);
            return;
        }

        const error = new Error('UNSUPPORTED_AUDIO_TYPE');
        error.code = 'UNSUPPORTED_AUDIO_TYPE';
        cb(error);
    }
});

// 연결 재사용(Keep-Alive) 에이전트 설정
const httpAgent = new http.Agent({ keepAlive: true, keepAliveMsecs: 1000 });
const httpsAgent = new https.Agent({ keepAlive: true, keepAliveMsecs: 1000 });

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));
app.use(express.static('public'));

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

class AppError extends Error {
    constructor(status, message, logMessage = message) {
        super(logMessage);
        this.status = status;
        this.publicMessage = message;
    }
}

function sendError(res, error, fallbackMessage) {
    if (res.headersSent) return;

    const status = error instanceof AppError ? error.status : 500;
    const message = error instanceof AppError ? error.publicMessage : fallbackMessage;

    res.status(status).json({ error: message });
}

function normalizeServerError(error) {
    if (error instanceof AppError) return error;

    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return new AppError(413, "파일이 너무 큽니다. 50MB 이하의 음원 파일을 선택해주세요.", error.message);
        }

        if (error.code === 'LIMIT_FILE_COUNT') {
            return new AppError(400, "한 번에 하나의 음원 파일만 업로드할 수 있습니다.", error.message);
        }

        return new AppError(400, "음원 파일 업로드에 실패했습니다.", error.message);
    }

    if (error.code === 'UNSUPPORTED_AUDIO_TYPE') {
        return new AppError(415, "오디오 파일만 업로드할 수 있습니다.");
    }

    if (error.type === 'entity.too.large') {
        return new AppError(413, "요청 데이터가 너무 큽니다.");
    }

    if (error instanceof SyntaxError && 'body' in error) {
        return new AppError(400, "요청 JSON 형식이 올바르지 않습니다.", error.message);
    }

    return new AppError(500, "서버 처리 중 오류가 발생했습니다.", error.message || String(error));
}

function parseNumber(value, fieldName) {
    const parsed = Number(value);

    if (!Number.isFinite(parsed)) {
        throw new AppError(400, "분석 요청 값이 올바르지 않습니다.", `${fieldName} must be a finite number`);
    }

    return parsed;
}

function validateAnalyzeRequest(req) {
    if (!req.file) {
        throw new AppError(400, "분석할 오디오 파일이 없습니다.");
    }

    const startTime = parseNumber(req.body.start_time, 'start_time');
    const duration = parseNumber(req.body.duration, 'duration');

    if (startTime < 0 || duration <= 0 || duration > 10) {
        throw new AppError(400, "분석 구간 값이 올바르지 않습니다.");
    }

    return { startTime, duration };
}

function validateAnalyzeResult(result) {
    if (!result || typeof result !== 'object' || Array.isArray(result)) {
        throw new AppError(502, "분석 서버 응답이 올바르지 않습니다.", "FastAPI response is not an object");
    }

    const entries = Object.entries(result);
    if (entries.length === 0) {
        throw new AppError(502, "분석 결과가 비어 있습니다.", "FastAPI response has no genre scores");
    }

    for (const [genre, score] of entries) {
        if (!genre || typeof genre !== 'string' || !Number.isFinite(score)) {
            throw new AppError(502, "분석 서버 응답이 올바르지 않습니다.", "FastAPI response has invalid genre score");
        }
    }
}

function validateGenres(genres) {
    if (!Array.isArray(genres) || genres.length === 0) {
        throw new AppError(400, "추천을 위한 장르 데이터가 없습니다.");
    }

    return genres.slice(0, 5).map((item) => {
        if (!item || typeof item !== 'object' || typeof item.genre !== 'string' || typeof item.ratio !== 'string') {
            throw new AppError(400, "추천 요청의 장르 데이터가 올바르지 않습니다.");
        }

        const genre = item.genre.trim();
        const ratio = item.ratio.trim();
        if (!genre || !ratio) {
            throw new AppError(400, "추천 요청의 장르 데이터가 올바르지 않습니다.");
        }

        return {
            genre,
            ratio
        };
    });
}

function validateRecommendations(recommendations) {
    if (!Array.isArray(recommendations) || recommendations.length === 0) {
        throw new AppError(502, "추천 결과가 비어 있습니다.", "Gemini response has no recommendations");
    }

    if (recommendations.length < 3) {
        throw new AppError(502, "추천 결과가 부족합니다.", "Gemini returned fewer than 3 recommendations");
    }

    return recommendations.slice(0, 3).map((item) => {
        if (
            !item ||
            typeof item !== 'object' ||
            typeof item.title !== 'string' ||
            typeof item.artist !== 'string' ||
            typeof item.reason !== 'string'
        ) {
            throw new AppError(502, "추천 결과 형식이 올바르지 않습니다.", "Gemini response has invalid recommendation item");
        }

        const title = item.title.trim();
        const artist = item.artist.trim();
        const reason = item.reason.trim();

        if (!title || !artist || !reason) {
            throw new AppError(502, "추천 결과 형식이 올바르지 않습니다.", "Gemini response has empty recommendation field");
        }

        return { title, artist, reason };
    });
}

async function fetchJsonWithTimeout(url, options, timeoutMs) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);

    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal
        });

        const responseText = await response.text();
        if (!response.ok) {
            let upstreamMessage = "분석 서버에서 오류가 발생했습니다.";

            try {
                const parsedError = JSON.parse(responseText);
                upstreamMessage = parsedError.error || parsedError.detail || upstreamMessage;
            } catch (error) {
                // Keep the generic public message when the upstream error body is not JSON.
            }

            const status = response.status >= 400 && response.status < 500 ? response.status : 502;
            throw new AppError(status, upstreamMessage, `FastAPI returned ${response.status}: ${responseText}`);
        }

        try {
            return JSON.parse(responseText);
        } catch (error) {
            throw new AppError(502, "분석 서버 응답을 해석하지 못했습니다.", `FastAPI returned invalid JSON: ${responseText}`);
        }
    } catch (error) {
        if (error instanceof AppError) {
            throw error;
        }

        if (error.name === 'AbortError') {
            throw new AppError(504, "분석 서버 응답 시간이 초과되었습니다.");
        }

        throw new AppError(502, "분석 서버에 연결하지 못했습니다.", error.message || "FastAPI connection failed");
    } finally {
        clearTimeout(timeout);
    }
}

async function withTimeout(promise, timeoutMs, publicMessage) {
    let timeout;

    try {
        return await Promise.race([
            promise,
            new Promise((_, reject) => {
                timeout = setTimeout(() => reject(new AppError(504, publicMessage)), timeoutMs);
            })
        ]);
    } finally {
        clearTimeout(timeout);
    }
}

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

    try {
        return JSON.parse(withoutFence);
    } catch (error) {
        throw new AppError(502, "AI 바리스타 응답을 해석하지 못했습니다.", `Gemini returned invalid JSON: ${text}`);
    }
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
    const filePath = req.file ? path.resolve(req.file.path) : null;
    let timerStarted = false;

    console.log(`[Node.js] 오디오 파일 도착, 분석을 요청합니다.`);

    try {
        const { startTime, duration } = validateAnalyzeRequest(req);

        console.time('FastAPI-Latency');
        timerStarted = true;
        
        const result = await fetchJsonWithTimeout(PYTHON_ANALYZE_URL, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Connection': 'keep-alive' 
            },
            agent: httpAgent,
            body: JSON.stringify({
                filepath: String(filePath),
                start_time: startTime,
                duration
            })
        }, PYTHON_TIMEOUT_MS);

        validateAnalyzeResult(result);
        console.timeEnd('FastAPI-Latency');
        timerStarted = false;
        
        console.log('[Node.js] 분석 완료! 손님에게 테이스팅 노트를 전달합니다.');
        res.json(result);

    } catch (error) {
        if (timerStarted) {
            console.timeEnd('FastAPI-Latency');
        }
        console.error("[Node.js 통신 에러]:", error.message);
        sendError(res, error, "오디오 분석 중 내부 통신 에러가 발생했습니다.");
    } finally {
        if (filePath) {
            try {
                await fs.unlink(filePath);
                console.log('[Node.js] 다 쓴 임시 오디오 파일을 깨끗하게 청소했습니다.');
            } catch (e) {
                console.error("[Node.js] 파일 청소 실패:", e);
            }
        }
    }
});

app.post('/api/recommend', async (req, res) => {
    const { genres, trackName } = req.body;
    const requestId = Date.now();
    const timerLabel = `Gemini-Latency-${requestId}`;
    let timerStarted = false;
    
    console.log(`[Node.js] 테이스팅 노트 도착! Gemini 바리스타에게 추천 곡을 묻습니다.`);

    try {
        if (!process.env.GEMINI_API_KEY) {
            throw new AppError(500, "추천 서비스 설정이 올바르지 않습니다.", "GEMINI_API_KEY is missing");
        }

        const safeGenres = validateGenres(genres);
        console.time(timerLabel);
        timerStarted = true;
        const model = genAI.getGenerativeModel({
            model: GEMINI_MODEL,
            generationConfig: {
                responseMimeType: "application/json"
            }
        });
        
        const prompt = `당신은 'Café de Music'의 친절하고 감성적인 AI 바리스타입니다. 손님이 다음 음악 장르 비율(테이스팅 노트)을 가진 음악을 들려주었습니다.
업로드한 파일명: ${trackName || '알 수 없음'}
장르 데이터: ${JSON.stringify(safeGenres)}

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

        const result = await withTimeout(
            model.generateContent(prompt),
            GEMINI_TIMEOUT_MS,
            "AI 바리스타 응답 시간이 초과되었습니다."
        );
        const response = await result.response;
        const text = response.text();
        const parsed = parseJsonResponse(text);
        const safeRecommendations = validateRecommendations(parsed.recommendations);
        const formattedRecommendation = formatRecommendations(safeRecommendations);

        console.timeEnd(timerLabel);
        timerStarted = false;
        console.log('[Node.js] AI 바리스타 추천 완료!');
        
        res.json({ recommendation: formattedRecommendation });
    } catch (error) {
        if (timerStarted) {
            console.timeEnd(timerLabel);
        }

        console.error("[Gemini 에러]:", error.message || error);
        sendError(
            res,
            error instanceof AppError ? error : new AppError(502, "AI 바리스타 호출 중 오류가 발생했습니다.", error.message || String(error)),
            "AI 바리스타의 큐레이션 생성에 실패했습니다."
        );
    }
});

app.use((error, req, res, next) => {
    const normalizedError = normalizeServerError(error);
    console.error("[Express 에러]:", normalizedError.message);
    sendError(res, normalizedError, "서버 처리 중 오류가 발생했습니다.");
});

const server = app.listen(3000, '127.0.0.1', () => {
    console.log('✨ [Node.js] 메인 서버 기동');
});

server.on('error', (error) => {
    if (error.code === 'EADDRINUSE') {
        console.error('[Node.js] 3000 포트가 이미 사용 중입니다. 기존 서버를 종료하거나 다른 포트를 사용해주세요.');
        return;
    }

    console.error('[Node.js] 서버 시작 중 오류가 발생했습니다:', error);
});
