import {
  CVAnalysisResultSchema,
  type CVAnalysisResult,
} from "@/lib/cv/validators/cv-analysis";
import OpenAI from "openai";

export class AIAnalysisError extends Error {
  constructor(
    message: string,
    public code:
      | "API_KEY_MISSING"
      | "AI_REQUEST_FAILED"
      | "INVALID_RESPONSE_SCHEMA"
  ) {
    super(message);
    this.name = "AIAnalysisError";
  }
}

const aiClient = new OpenAI({
  apiKey: process.env.GEMINI_API_KEY || "",
  baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/",
  maxRetries: 3,
  timeout: 60000,
});

// ── Giới hạn độ dài text gửi AI (~3000 tokens) ───────────────────────────────
const MAX_CV_CHARS = 12_000;

// ── Tiền xử lý text CV trước khi gửi AI ─────────────────────────────────────
function preprocessCVText(text: string): string {
  return (
    text
      // Loại bỏ dòng chỉ chứa số trang
      .replace(/^(page\s*\d+|\d+\s*of\s*\d+)$/gim, "")
      // Gộp từ bị tách dòng ngắn (hay gặp ở PDF 2 cột)
      .replace(/([a-z,])\n([a-z])/g, "$1 $2")
      // Rút gọn URL quá dài
      .replace(/https?:\/\/\S{80,}/g, "[URL]")
      // Loại bỏ dòng trắng thừa
      .replace(/\n{3,}/g, "\n\n")
      .trim()
  );
}

// ── System Prompt ─────────────────────────────────────────────────────────────
const CV_ANALYSIS_SYSTEM_PROMPT = `
You are an expert HR Specialist and Technical Recruiter specializing in the Vietnamese IT job market.

Your first task is to determine whether the provided document is actually a Curriculum Vitae (CV) / Resume of an individual.

=== MARKET CONTEXT ===
You are evaluating CVs for the Vietnamese tech job market:
- Vietnamese CVs may lack a GitHub profile or portfolio link (acceptable for juniors/freshers).
- CVs may mix Vietnamese and English — this is normal and acceptable.
- Certifications from Vietnamese training centers (FPT, Aptech, VNPT, etc.) are valid credentials.
- Score the CV relative to the Vietnamese market standard, not Silicon Valley.
- Write all assessmentSummary, improvements, and strengths in Vietnamese.

=== CRITICAL PROJECT EXTRACTION INSTRUCTION ===
For students, interns, and junior developers in Vietnam, PROJECTS (Đồ án tốt nghiệp, bài tập lớn, đồ án môn học, dự án cá nhân, capstone projects) are the single most important evidence of capability.
- You MUST thoroughly scan the CV and extract EVERY single project into the "projects" array.
- Look for sections named: "DỰ ÁN", "PROJECTS", "PERSONAL PROJECTS", "ACADEMIC PROJECTS", "ĐỒ ÁN", or projects mentioned within work experience.
- For each project:
  * "name": Full name of the project (e.g. "Hệ thống RAG Pipeline", "Website E-Commerce")
  * "description": Key features, responsibilities, or impact
  * "technologies": Array of technologies/tools used (e.g. ["Vue 3", "Spring Boot", "MySQL", "Docker"])
  * "url": GitHub link or live demo URL, or null if none is specified.
- Do NOT skip projects. If there are 3 projects in the CV, extract all 3.

=== SCORING RUBRIC (0–100) ===
- 90–100: Xuất sắc — CV rất chuyên nghiệp, thành tích định lượng rõ ràng, phù hợp vị trí senior.
- 75–89: Tốt — CV đầy đủ, còn 1–2 điểm thiếu nhỏ, phù hợp mid-level.
- 60–74: Khá — Đủ nội dung cơ bản nhưng cần bổ sung chiều sâu hoặc cấu trúc.
- 40–59: Trung bình — Thiếu nhiều mục quan trọng hoặc nội dung quá sơ sài.
- 0–39: Yếu — CV kém chất lượng hoặc không phải CV.

Điểm trừ:
- Thiếu thông tin liên hệ: -10 điểm
- Không có số liệu định lượng trong kinh nghiệm: -15 điểm
- Cấu trúc lộn xộn, khó đọc: -10 điểm
- Kỹ năng liệt kê không có ngữ cảnh minh chứng: -5 điểm

=== OUTPUT FORMAT ===
Respond STRICTLY with a valid JSON object matching this EXACT template (no markdown, no extra text):

{
  "isCV": true,
  "rejectionReason": null,
  "personalInfo": {
    "fullName": "Họ tên hoặc null",
    "email": "email hoặc null",
    "phone": "số điện thoại hoặc null",
    "summary": "Tóm tắt 1-2 câu về ứng viên hoặc null"
  },
  "experienceLevel": "junior",
  "yearsOfExperience": 2,
  "skills": {
    "technical": ["React", "Node.js", "PostgreSQL"],
    "soft": ["Giao tiếp", "Làm việc nhóm"]
  },
  "experience": [
    {
      "company": "Tên công ty",
      "role": "Chức danh",
      "duration": "2022 - 2024",
      "highlights": ["Thành tích 1", "Thành tích 2"]
    }
  ],
  "education": [
    {
      "institution": "Tên trường",
      "degree": "Cử nhân",
      "major": "Công nghệ Thông tin",
      "graduationYear": "2023"
    }
  ],
  "certifications": [
    {
      "name": "Tên chứng chỉ",
      "issuer": "Tổ chức cấp",
      "year": "2023"
    }
  ],
  "projects": [
    {
      "name": "Tên dự án",
      "description": "Mô tả ngắn",
      "technologies": ["React", "Node.js"],
      "url": "https://github.com/... hoặc null"
    }
  ],
  "strengths": ["Điểm mạnh 1", "Điểm mạnh 2", "Điểm mạnh 3"],
  "improvements": ["Cần cải thiện 1 (cụ thể)", "Cần cải thiện 2 (cụ thể)"],
  "score": 72,
  "assessmentSummary": "Nhận xét tổng quan bằng tiếng Việt về ứng viên này."
}

=== GUARDRAIL RULES ===
If the document is NOT a CV (e.g. research paper, book, invoice, project report, exam, source code, legal contract):
- Set "isCV": false
- Set "rejectionReason": "Giải thích bằng tiếng Việt tại sao tài liệu này không phải CV."
- Set "score": 0
- Set "experienceLevel": "fresher"
- Set "yearsOfExperience": null
- Set all arrays to []
`;

// ── Main function ─────────────────────────────────────────────────────────────
export async function analyzeCVWithAI(
  extractedText: string
): Promise<CVAnalysisResult> {
  if (!process.env.GEMINI_API_KEY) {
    throw new AIAnalysisError(
      "GEMINI_API_KEY is not configured in the environment.",
      "API_KEY_MISSING"
    );
  }

  // 1. Tiền xử lý và cắt ngắn text
  const processedText = preprocessCVText(extractedText);
  const truncatedText =
    processedText.length > MAX_CV_CHARS
      ? processedText.slice(0, MAX_CV_CHARS) +
        "\n\n[... nội dung bị rút gọn để phân tích ...]"
      : processedText;

  let rawResponse = "";
  try {
    const res = await aiClient.chat.completions.create({
      model: process.env.GEMINI_MODEL || "gemini-3.5-flash",
      messages: [
        { role: "system", content: CV_ANALYSIS_SYSTEM_PROMPT },
        {
          role: "user",
          content: `Dưới đây là nội dung CV được trích xuất:\n\n"""\n${truncatedText}\n"""\n\nPhân tích CV này và trả về JSON theo đúng template đã yêu cầu.`,
        },
      ],
      response_format: { type: "json_object" },
      temperature: 0.1, // Tối thiểu hoá sự ngẫu nhiên — ưu tiên nhất quán
    });

    rawResponse = res.choices[0]?.message?.content || "";
  } catch (err: unknown) {
    throw new AIAnalysisError(
      `Gemini request failed: ${(err as Error).message}`,
      "AI_REQUEST_FAILED"
    );
  }

  try {
    const cleanedJson = rawResponse
      .replace(/^```json\s*/i, "")
      .replace(/^```\s*/i, "")
      .replace(/\s*```$/i, "")
      .trim();

    const parsedJson = JSON.parse(cleanedJson);

    // Dùng safeParse để log chi tiết field nào bị lỗi trước khi throw
    const result = CVAnalysisResultSchema.safeParse(parsedJson);
    if (!result.success) {
      console.error("--- RAW AI RESPONSE ---", rawResponse);
      console.error(
        "--- ZOD FIELD ERRORS ---",
        JSON.stringify(result.error.flatten(), null, 2)
      );
      throw new AIAnalysisError(
        "AI response did not adhere to the required JSON schema.",
        "INVALID_RESPONSE_SCHEMA"
      );
    }
    return result.data;
  } catch (err: unknown) {
    if (err instanceof AIAnalysisError) throw err;
    // Lỗi JSON.parse
    console.error("--- RAW AI RESPONSE (JSON parse failed) ---", rawResponse);
    console.error("--- ERROR ---", err);
    throw new AIAnalysisError(
      "AI response did not adhere to the required JSON schema.",
      "INVALID_RESPONSE_SCHEMA"
    );
  }
}
