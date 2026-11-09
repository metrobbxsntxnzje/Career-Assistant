export const runtime = "nodejs";
import { Prisma } from "@/app/generated/prisma";
import { AIAnalysisError, analyzeCVWithAI } from "@/lib/ai/cv-analyzer";
import { getCurrentUser } from "@/lib/auth";
import { CVParserError, extractTextFormFile } from "@/lib/cv/cv-parser";
import { prisma } from "@/lib/prisma";
import { apiError } from "@/lib/utils";
import { NextResponse } from "next/server";

export async function GET(request: Request) {
  const user = await getCurrentUser();
  if (!user)
    return apiError(401, "UNAUTHENTICATED", "Please sign in to continue");

  const { searchParams } = new URL(request.url);
  const cvId = searchParams.get("cvId");
  try {
    const whereCondition = cvId
      ? { id: cvId, userId: user.id }
      : { userId: user.id, status: "ANALYZED" };
    const cv = await prisma.cV.findFirst({
      where: whereCondition,
      orderBy: {
        createdAt: "desc",
      },
      select: {
        id: true,
        originalName: true,
        status: true,
        score: true,
        extractedText: true,
        analysisJson: true,
        createdAt: true,
        updatedAt: true,
      },
    });

    if (!cv) {
      return NextResponse.json({ data: null });
    }

    return NextResponse.json({ data: cv });
  } catch (error: unknown) {
    console.error("GET CV error:", error);
    return apiError(500, "INTERNAL_ERROR", "Failed to retrieve CV data.");
  }
}

export async function POST(request: Request) {
  const currentUser = await getCurrentUser();
  if (!currentUser) {
    return apiError(401, "UNAUTHENTICATED", "Please sign in to continue");
  }

  let cvId: string | undefined;
  try {
    const body = await request.json().catch(() => ({}));
    cvId = body.cvId;
    if (!cvId) {
      return NextResponse.json(
        { message: "Missing required field 'cvId'.", code: "INVALID_REQUEST" },
        { status: 400 }
      );
    }

    const cv = await prisma.cV.findFirst({
      where: { id: cvId, userId: currentUser.id },
    });
    if (!cv) {
      return NextResponse.json(
        { message: "CV not found with the provided ID.", code: "CV_NOT_FOUND" },
        { status: 404 }
      );
    }

    // ── Bước 1: Parse text từ S3 ─────────────────────────────────────────────
    let cleanText = "";
    try {
      cleanText = await extractTextFormFile(cv.filePath);
    } catch (parseError) {
      await prisma.cV.update({
        where: { id: cvId },
        data: { status: "FAILED" },
      });
      if (parseError instanceof CVParserError) {
        return NextResponse.json(
          { message: parseError.message, code: parseError.code },
          { status: 422 }
        );
      }
      throw parseError;
    }

    await prisma.cV.update({
      where: { id: cvId },
      data: { extractedText: cleanText, status: "PARSED" },
    });

    // ── Bước 2: Gọi AI phân tích ─────────────────────────────────────────────
    let analysisResult: Awaited<ReturnType<typeof analyzeCVWithAI>>;
    try {
      analysisResult = await analyzeCVWithAI(cleanText);
    } catch (aiError) {
      await prisma.cV.update({
        where: { id: cvId },
        data: { status: "FAILED" },
      });
      if (aiError instanceof AIAnalysisError) {
        return NextResponse.json(
          { message: aiError.message, code: aiError.code },
          { status: 502 }
        );
      }
      throw aiError;
    }

    // ── Bước 3: Lưu kết quả phân tích ───────────────────────────────────────
    const updatedCV = await prisma.cV.update({
      where: { id: cvId },
      data: {
        analysisJson: analysisResult as Prisma.InputJsonValue,
        score: analysisResult.score,
        status: "ANALYZED",
      },
      select: {
        id: true,
        originalName: true,
        status: true,
        score: true,
        extractedText: true,
        analysisJson: true,
        updatedAt: true,
      },
    });

    // ── Bước 4: Lưu Skills vào DB (Deduplicated & Non-blocking) ─────────────
    try {
      const allSkills: { name: string; category: "technical" | "soft" }[] = [
        ...analysisResult.skills.technical.map((name) => ({
          name: name.trim(),
          category: "technical" as const,
        })),
        ...analysisResult.skills.soft.map((name) => ({
          name: name.trim(),
          category: "soft" as const,
        })),
      ].filter((s) => s.name.length > 0);

      // Loại bỏ trùng lặp tên skill
      const uniqueSkillMap = new Map<string, "technical" | "soft">();
      for (const item of allSkills) {
        const lower = item.name.toLowerCase();
        if (!uniqueSkillMap.has(lower)) {
          uniqueSkillMap.set(lower, item.category);
        }
      }

      // Xử lý từng skill không dùng interactive transaction để tránh timeout trên cloud serverless
      for (const [lowerName, category] of uniqueSkillMap.entries()) {
        try {
          let skill = await prisma.skill.findFirst({
            where: { name: { equals: lowerName, mode: "insensitive" } },
          });

          if (!skill) {
            const originalName =
              allSkills.find((s) => s.name.toLowerCase() === lowerName)?.name || lowerName;
            skill = await prisma.skill.create({
              data: { name: originalName, category },
            });
          }

          const existingLink = await prisma.cVSkill.findFirst({
            where: { cvId: cv.id, skillId: skill.id },
          });

          if (!existingLink) {
            await prisma.cVSkill.create({
              data: { cvId: cv.id, skillId: skill.id },
            });
          }
        } catch (itemErr) {
          // Bỏ qua lỗi 1 skill lẻ, không làm gián đoạn
          console.warn(`[cv/analyze] Failed to link skill "${lowerName}":`, itemErr);
        }
      }
    } catch (skillSyncError) {
      console.warn("[cv/analyze] Non-fatal error during skill sync:", skillSyncError);
    }

    return NextResponse.json({
      message: "CV parsed and analyzed successfully.",
      data: updatedCV,
    });
  } catch (error: unknown) {
    console.error("CV analyze pipeline error:", error);

    if (cvId) {
      await prisma.cV
        .update({
          where: { id: cvId },
          data: { status: "FAILED" },
        })
        .catch(() => null);
    }

    return NextResponse.json(
      {
        message: "An unexpected error occurred during CV analysis.",
        error: (error as Error).message,
      },
      { status: 500 }
    );
  }
}
