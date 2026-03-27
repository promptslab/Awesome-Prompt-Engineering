import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  let email = "";
  let identity = "";

  try {
    const body = await request.json();
    email = body.email;
    identity = body.identity;
  } catch (err) {
    console.error("[Subscribe] Failed to parse request body:", err);
    return NextResponse.json(
      { error: "Invalid request body" },
      { status: 400 }
    );
  }

  if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return NextResponse.json(
      { error: "Invalid email address" },
      { status: 400 }
    );
  }

  if (!identity) {
    return NextResponse.json(
      { error: "Identity is required" },
      { status: 400 }
    );
  }

  // Log every signup regardless
  console.log(`[Subscribe] ${email} (${identity}) — ${new Date().toISOString()}`);

  const apiKey = process.env.RESEND_API_KEY;
  const notifyEmail = process.env.NOTIFY_EMAIL;

  if (!apiKey || !notifyEmail) {
    return NextResponse.json({ success: true });
  }

  try {
    const { Resend } = await import("resend");
    const resend = new Resend(apiKey);

    const { error } = await resend.emails.send({
      from: "PromptsLab Course <onboarding@resend.dev>",
      to: notifyEmail,
      subject: `New Course Signup: ${email}`,
      html: `
        <h2>New Course Waitlist Signup</h2>
        <table style="border-collapse:collapse;font-family:sans-serif;">
          <tr><td style="padding:8px;font-weight:bold;">Email</td><td style="padding:8px;">${email}</td></tr>
          <tr><td style="padding:8px;font-weight:bold;">Role</td><td style="padding:8px;">${identity}</td></tr>
          <tr><td style="padding:8px;font-weight:bold;">Time</td><td style="padding:8px;">${new Date().toISOString()}</td></tr>
        </table>
      `,
    });

    if (error) {
      console.error("[Subscribe] Resend error:", error);
    }
  } catch (err) {
    console.error("[Subscribe] Email send failed:", err);
  }

  // Always return success — signup is logged even if email fails
  return NextResponse.json({ success: true });
}
