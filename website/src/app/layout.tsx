import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "PromptsLab — Awesome Prompt Engineering",
    template: "%s | PromptsLab",
  },
  description:
    "A hand-curated collection of resources for Prompt Engineering and Context Engineering — papers, tools, models, APIs, benchmarks, courses, and communities. By PromptsLab.",
  openGraph: {
    title: "PromptsLab — Awesome Prompt Engineering",
    description:
      "A hand-curated collection of resources for Prompt Engineering and Context Engineering. By PromptsLab.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} min-h-screen font-sans antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
