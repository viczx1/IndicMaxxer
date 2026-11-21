import type { Metadata } from "next";
import { Exo_2, Noto_Sans, Share_Tech_Mono, Noto_Sans_Kannada, Noto_Sans_Devanagari } from "next/font/google";
import SmoothScroll from "@/components/SmoothScroll";
import "./globals.css";

// Display: Exo 2 (2000s tech, PS2-era feel). Body: Segoe UI on Windows (the Vista face), Noto Sans elsewhere.
const display = Exo_2({ subsets: ["latin"], variable: "--font-display", display: "swap", weight: ["300", "400", "500", "600", "700"] });
const body = Noto_Sans({ subsets: ["latin"], variable: "--font-body", display: "swap", weight: ["400", "500", "600"] });
const mono = Share_Tech_Mono({ subsets: ["latin"], variable: "--font-mono-x", display: "swap", weight: "400" });
const kn = Noto_Sans_Kannada({ subsets: ["kannada"], variable: "--font-kn", display: "swap", weight: ["400", "600"] });
const dev = Noto_Sans_Devanagari({ subsets: ["devanagari"], variable: "--font-dev", display: "swap", weight: ["400", "600"] });

export const metadata: Metadata = {
  title: "IndicMaxxer-2B",
  description: "A local Indian-language model that cuts the token tax. Kannada, Hindi, English, Kanglish, Hinglish.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${display.variable} ${body.variable} ${mono.variable} ${kn.variable} ${dev.variable} indic`}>
        <SmoothScroll />
        {children}
      </body>
    </html>
  );
}
