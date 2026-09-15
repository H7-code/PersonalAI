import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "ARIA — Adaptive Real-time Intelligent Assistant",
  description: "Fully offline, multilingual voice assistant powered by DeepSeek-R1, Faster-Whisper, Piper & MMS-TTS.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full bg-slate-50">
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} h-full font-sans antialiased text-slate-800 bg-radial-luminous flex flex-col justify-between overflow-x-hidden selection:bg-emerald-100 selection:text-emerald-900`}
      >
        {children}
      </body>
    </html>
  );
}
