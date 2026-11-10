"""UI Theme, styles, and HTML snippets for Clinical Trial Eligibility Screening.
Matches the soft, modern, 'bento-grid' healthcare landing page aesthetic.
"""
import gradio as gr

def get_theme() -> gr.themes.Base:
    """Soft, modern light theme with Inter typography and large rounded corners."""
    theme = gr.themes.Base(
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
        font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas", "monospace"],
    ).set(
        body_background_fill="#E6E9F4",
        body_background_fill_dark="#E6E9F4",
        body_text_color="#111827",
        body_text_color_dark="#111827",
        body_text_color_subdued="#64748B",
        body_text_color_subdued_dark="#64748B",
        
        background_fill_primary="#FFFFFF",
        background_fill_primary_dark="#FFFFFF",
        background_fill_secondary="#F8FAFC",
        background_fill_secondary_dark="#F8FAFC",
        
        block_background_fill="#FFFFFF",
        block_background_fill_dark="#FFFFFF",
        block_border_color="#E2E8F0",
        block_border_color_dark="#E2E8F0",
        block_border_width="1px",
        block_radius="20px",
        
        border_color_primary="#E2E8F0",
        border_color_primary_dark="#E2E8F0",
        color_accent="#5B9FFF",
        color_accent_soft="#E8F1FF",
        
        button_primary_background_fill="#0A0A0A",
        button_primary_background_fill_dark="#0A0A0A",
        button_primary_background_fill_hover="#262626",
        button_primary_background_fill_hover_dark="#262626",
        button_primary_text_color="#FFFFFF",
        button_primary_text_color_dark="#FFFFFF",
        button_primary_border_color="#0A0A0A",
        button_primary_border_color_dark="#0A0A0A",
        button_large_radius="9999px",
        button_medium_radius="9999px",
        button_small_radius="9999px",
        
        input_background_fill="#FFFFFF",
        input_background_fill_dark="#FFFFFF",
        input_border_color="#E2E8F0",
        input_border_color_dark="#E2E8F0",
        input_border_width="1.5px",
        input_radius="16px",
        
        table_border_color="#E2E8F0",
        table_border_color_dark="#E2E8F0",
        table_even_background_fill="#FAFAFA",
        table_odd_background_fill="#FFFFFF",
        table_row_focus="#F0F5FF",
    )
    return theme

def get_head_html() -> str:
    return """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<meta name="color-scheme" content="light">
"""

def get_medical_illustration_svg() -> str:
    """Original flat line-art medical illustration (black outlines, blue accents) + curly-arrow doodle."""
    return """
<div class="hero-art-wrapper">
<svg width="220" height="175" viewBox="0 0 300 240" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="Medical consultation illustration">
  <!-- Decorative hand-drawn curly-arrow doodle -->
  <path d="M 245,35 C 285,65 260,115 220,135 C 195,150 200,190 245,170 C 280,155 260,85 240,95 C 220,105 190,175 160,170" 
        stroke="#1A1A1A" stroke-width="2.2" stroke-linecap="round" fill="none"/>
  <!-- Doodle arrow tip -->
  <path d="M 168,165 L 158,170 L 166,177" stroke="#1A1A1A" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>

  <!-- Computer monitor in background -->
  <rect x="75" y="48" width="42" height="30" rx="4" stroke="#94A3B8" stroke-width="2" fill="#FFFFFF"/>
  <line x1="96" y1="78" x2="96" y2="88" stroke="#94A3B8" stroke-width="2"/>
  <line x1="86" y1="88" x2="106" y2="88" stroke="#94A3B8" stroke-width="2"/>
  <line x1="80" y1="56" x2="112" y2="56" stroke="#CBD5E1" stroke-width="2"/>
  <line x1="80" y1="62" x2="102" y2="62" stroke="#CBD5E1" stroke-width="2"/>

  <!-- Patient (left, sitting on stool) -->
  <ellipse cx="52" cy="148" rx="18" ry="5" fill="#E2E8F0" stroke="#1A1A1A" stroke-width="2.2"/>
  <line x1="52" y1="153" x2="52" y2="178" stroke="#1A1A1A" stroke-width="2.2"/>
  <line x1="52" y1="178" x2="38" y2="188" stroke="#1A1A1A" stroke-width="2.2"/>
  <line x1="52" y1="178" x2="66" y2="188" stroke="#1A1A1A" stroke-width="2.2"/>

  <!-- Patient Head & Hair -->
  <ellipse cx="55" cy="58" rx="13" ry="15" fill="#FFFFFF" stroke="#1A1A1A" stroke-width="2.2"/>
  <path d="M 44,58 C 43,46 52,43 64,46 C 68,47 69,53 69,56 C 66,54 62,52 56,54 C 49,56 46,61 44,58 Z" fill="#475569"/>
  <!-- Patient Torso (Cornflower Blue) -->
  <path d="M 44,75 C 44,71 63,71 66,75 L 68,115 C 68,118 42,118 42,115 Z" fill="#5B9FFF" stroke="#1A1A1A" stroke-width="2.2"/>
  <!-- Patient Arm -->
  <path d="M 45,76 L 37,100 L 52,106" fill="none" stroke="#1A1A1A" stroke-width="2.2" stroke-linecap="round"/>
  <!-- Patient Pants -->
  <path d="M 43,115 C 43,115 67,115 67,117 L 63,144 L 53,144 L 49,170 L 40,170 L 43,115 Z" fill="#FFFFFF" stroke="#1A1A1A" stroke-width="2.2"/>
  <path d="M 39,170 C 39,170 34,170 34,175 C 34,177 48,177 48,175 Z" fill="#1A1A1A"/>

  <!-- Doctor (right, sitting on clinic chair) -->
  <path d="M 152,85 L 152,128" stroke="#1A1A1A" stroke-width="2.2"/>
  <path d="M 152,128 L 128,131" stroke="#1A1A1A" stroke-width="2.2"/>
  <line x1="138" y1="131" x2="138" y2="168" stroke="#1A1A1A" stroke-width="2.2"/>
  <line x1="138" y1="168" x2="120" y2="178" stroke="#1A1A1A" stroke-width="2.2"/>
  <line x1="138" y1="168" x2="155" y2="178" stroke="#1A1A1A" stroke-width="2.2"/>

  <!-- Doctor Head -->
  <ellipse cx="138" cy="50" rx="13" ry="14" fill="#FFFFFF" stroke="#1A1A1A" stroke-width="2.2"/>
  <!-- Spectacles -->
  <circle cx="134" cy="50" r="3.5" fill="none" stroke="#1A1A1A" stroke-width="2"/>
  <circle cx="143" cy="50" r="3.5" fill="none" stroke="#1A1A1A" stroke-width="2"/>
  <line x1="137.5" y1="50" x2="139.5" y2="50" stroke="#1A1A1A" stroke-width="2"/>
  <!-- Hair -->
  <path d="M 126,48 C 126,37 138,35 148,39 C 151,41 151,49 150,49 C 148,46 143,44 136,45 C 130,46 127,51 126,48 Z" fill="#1E293B"/>

  <!-- Doctor Legs (Cornflower Blue trousers) -->
  <path d="M 124,108 L 140,108 L 138,138 L 122,138 Z" fill="#5B9FFF" stroke="#1A1A1A" stroke-width="2.2"/>
  <path d="M 122,138 L 122,168 L 112,168 L 112,138 Z" fill="#5B9FFF" stroke="#1A1A1A" stroke-width="2.2"/>
  <path d="M 110,168 C 110,168 104,168 104,173 C 104,175 120,175 120,173 Z" fill="#1A1A1A"/>

  <!-- White Lab Coat -->
  <path d="M 125,66 C 125,66 148,66 150,70 L 152,122 C 152,122 118,122 118,119 L 123,70 Z" fill="#FFFFFF" stroke="#1A1A1A" stroke-width="2.2"/>
  <!-- Stethoscope (Blue accent) -->
  <path d="M 132,67 L 136,79 L 142,67" fill="none" stroke="#5B9FFF" stroke-width="2.2"/>

  <!-- Doctor Gesturing Left Hand -->
  <path d="M 123,71 L 106,77 L 98,73" fill="none" stroke="#1A1A1A" stroke-width="2.2" stroke-linecap="round"/>
  <!-- Right Hand holding Medical Chart -->
  <rect x="138" y="73" width="16" height="24" rx="2" fill="#FFFFFF" stroke="#1A1A1A" stroke-width="2"/>
  <line x1="142" y1="72" x2="150" y2="72" stroke="#1A1A1A" stroke-width="2.2"/>
  <line x1="141" y1="79" x2="151" y2="79" stroke="#94A3B8" stroke-width="1.5"/>
  <line x1="141" y1="84" x2="149" y2="84" stroke="#94A3B8" stroke-width="1.5"/>
</svg>
</div>
"""

def get_blue_card_html(engine) -> str:
    """Cornflower blue card with real engine status."""
    device = getattr(engine, "device", "cpu")
    base = getattr(engine, "base", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter = getattr(engine, "adapter", None)
    adapter_label = "LoRA r16 fine-tuned" if adapter else "Base model (testing mode)"
    
    return f"""
<div class="card-blue-inner">
  <div class="badge-white-pill">74.3% accuracy on held-out test set</div>
  <div class="ecg-container">
    <svg class="ecg-svg" viewBox="0 0 320 80" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path d="M 0,50 L 40,50 L 52,25 L 64,75 L 82,8 L 98,68 L 110,40 L 122,50 L 155,50 L 170,18 L 188,78 L 202,32 L 216,50 L 320,50" 
            stroke="#FFFFFF" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
      <circle cx="170" cy="18" r="6.5" fill="#FFFFFF"/>
    </svg>
  </div>
  <div class="card-blue-bottom">
    <div class="card-blue-title">{adapter_label}</div>
    <div class="card-blue-sub">Model: {base}<br>Device: {device}</div>
  </div>
</div>
"""

def get_grey_card_header_html() -> str:
    """Header section for the big grey card with headline, subtitle, and illustration."""
    art_svg = get_medical_illustration_svg()
    return f"""
<div class="hero-banner-row">
  <div class="hero-text-col">
    <h1 class="card-grey-headline">Screen a patient against trial criteria</h1>
    <p class="card-grey-sub">
      Automated eligibility verification powered by fine-tuned instruction models.
      Parse clinical notes against multi-point protocol criteria with sentence-level citations.
    </p>
  </div>
  <div class="hero-art-col">
    {art_svg}
  </div>
</div>
"""

def get_custom_css() -> str:
    return """
/* === Force Light Mode & Base Theme === */
:root, html, body {
    color-scheme: light !important;
    background-color: #E6E9F4 !important;
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    margin: 0;
    padding: 0;
}

body {
    background-color: #E6E9F4 !important;
    color: #111827 !important;
}

/* Base Gradio Container */
.gradio-container {
    background-color: #E6E9F4 !important;
    max-width: 100% !important;
    width: 100% !important;
    min-height: 100vh !important;
    padding: 16px 20px !important;
    box-sizing: border-box !important;
}

@media (min-width: 768px) {
    .gradio-container {
        padding: 24px 32px !important;
    }
}

/* Screen-filling Bento Container */
.bento-shell {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 auto !important;
    background-color: #F8F8F8 !important;
    border-radius: 32px !important;
    padding: 28px 24px !important;
    box-shadow: 0 12px 48px rgba(0, 0, 0, 0.04) !important;
    border: 1px solid rgba(0, 0, 0, 0.04) !important;
    box-sizing: border-box !important;
}

@media (min-width: 768px) {
    .bento-shell {
        padding: 36px 44px !important;
    }
}

/* --- Safety Disclaimer Pill --- */
.disclaimer-bar {
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    background-color: #FFF3E0 !important;
    border: 1px solid #FFE0B2 !important;
    color: #9A3412 !important;
    padding: 10px 20px !important;
    border-radius: 9999px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    margin-bottom: 24px !important;
    line-height: 1.4 !important;
}
.disclaimer-bar strong {
    color: #C2410C !important;
    font-weight: 700 !important;
}

/* Testing Mode Alert Banner */
.test-mode-alert {
    background-color: #FEF2F2 !important;
    border: 1px solid #FECACA !important;
    color: #991B1B !important;
    padding: 10px 18px !important;
    border-radius: 16px !important;
    margin-bottom: 20px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* --- Top Navigation Bar --- */
.bento-top-bar {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    margin-bottom: 24px !important;
    padding-bottom: 4px !important;
    gap: 16px !important;
    flex-wrap: wrap !important;
}

.bento-wordmark {
    font-size: 1.45rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    color: #111827 !important;
    white-space: nowrap !important;
}

.nav-badge-pill {
    background-color: #262626 !important;
    color: #FFFFFF !important;
    padding: 8px 18px !important;
    border-radius: 9999px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    white-space: nowrap !important;
}

/* Navigation Tabs styled as centered nav links */
.bento-tabs > .tab-nav {
    display: flex !important;
    justify-content: center !important;
    border-bottom: none !important;
    gap: 20px !important;
    margin-bottom: 24px !important;
    background: transparent !important;
}

.bento-tabs > .tab-nav > button {
    border: none !important;
    background: transparent !important;
    color: #64748B !important;
    font-size: 1.02rem !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    border-radius: 9999px !important;
    transition: all 0.2s ease !important;
}

.bento-tabs > .tab-nav > button:hover {
    color: #0F172A !important;
    background: rgba(0, 0, 0, 0.04) !important;
}

.bento-tabs > .tab-nav > button.selected {
    color: #0F172A !important;
    font-weight: 700 !important;
    background: #ECEFF4 !important;
    border: none !important;
}

/* --- Bento Grid Hero Layout --- */
.bento-hero-row {
    display: flex !important;
    flex-direction: column !important;
    gap: 24px !important;
    margin-bottom: 24px !important;
}

@media (min-width: 900px) {
    .bento-hero-row {
        flex-direction: row !important;
        align-items: stretch !important;
    }
    .bento-col-left {
        flex: 0 0 380px !important;
        max-width: 420px !important;
    }
    .bento-col-right {
        flex: 1 1 0 !important;
        min-width: 0 !important;
    }
}

.bento-col-left {
    display: flex !important;
    flex-direction: column !important;
    gap: 24px !important;
}

/* === Bento Card 1: Cornflower Blue Card (~#5B9FFF, radius ~28px) === */
.card-blue {
    background: #5B9FFF !important;
    border-radius: 28px !important;
    padding: 28px !important;
    color: #FFFFFF !important;
    position: relative !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
    min-height: 270px !important;
    border: none !important;
}

.card-blue-inner {
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    height: 100% !important;
    min-height: 220px !important;
}

.card-blue .badge-white-pill {
    align-self: flex-start !important;
    background-color: #FFFFFF !important;
    color: #1E60D5 !important;
    padding: 6px 14px !important;
    border-radius: 9999px !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06) !important;
}

.card-blue .ecg-container {
    margin: 14px 0 !important;
}

.card-blue .ecg-svg {
    width: 100% !important;
    height: 75px !important;
}

.card-blue-title {
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    line-height: 1.25 !important;
    color: #FFFFFF !important;
    margin: 0 0 6px 0 !important;
}

.card-blue-sub {
    font-size: 0.84rem !important;
    color: rgba(255, 255, 255, 0.9) !important;
    line-height: 1.4 !important;
    margin: 0 !important;
}

/* === Bento Card 2: Near-Black Card (~#0A0A0A, radius ~28px) === */
.card-black {
    background-color: #0A0A0A !important;
    border-radius: 28px !important;
    padding: 26px !important;
    color: #FFFFFF !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 16px !important;
    box-sizing: border-box !important;
    border: none !important;
}

.black-card-title {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    margin: 0 0 4px 0 !important;
    letter-spacing: -0.02em !important;
}

.black-card-desc {
    font-size: 0.82rem !important;
    color: #9CA3AF !important;
    margin: 0 0 12px 0 !important;
    line-height: 1.4 !important;
}

.card-black label span {
    color: #E5E7EB !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

.card-black .gr-dropdown {
    background-color: #171717 !important;
    border: 1px solid #333333 !important;
    border-radius: 14px !important;
    color: #FFFFFF !important;
}
.card-black input {
    color: #FFFFFF !important;
}

/* Buttons inside Black Card */
.btn-white-outline {
    background: transparent !important;
    border: 1.5px solid #FFFFFF !important;
    color: #FFFFFF !important;
    border-radius: 9999px !important;
    padding: 8px 18px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
.btn-white-outline:hover {
    background: #FFFFFF !important;
    color: #0A0A0A !important;
}

.btn-black-clear {
    background: #1F1F1F !important;
    border: 1px solid #333333 !important;
    color: #D1D5DB !important;
    border-radius: 9999px !important;
    padding: 8px 18px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
.btn-black-clear:hover {
    background: #2D2D2D !important;
    color: #FFFFFF !important;
}

/* === Bento Card 3: Big Light-Grey Card (~#F2F2F2, radius ~32px) === */
.card-grey {
    background-color: #F2F2F2 !important;
    border-radius: 32px !important;
    padding: 28px 24px !important;
    display: flex !important;
    flex-direction: column !important;
    box-sizing: border-box !important;
    border: 1px solid rgba(0, 0, 0, 0.03) !important;
}

@media (min-width: 768px) {
    .card-grey {
        padding: 38px 36px !important;
    }
}

.hero-banner-row {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 16px !important;
    margin-bottom: 22px !important;
}

.hero-text-col {
    flex: 1 1 auto !important;
}

.hero-art-col {
    flex: 0 0 210px !important;
    text-align: right !important;
}

@media (max-width: 680px) {
    .hero-banner-row {
        flex-direction: column-reverse !important;
        align-items: flex-start !important;
    }
    .hero-art-col {
        align-self: center !important;
    }
}

.card-grey-headline {
    font-size: 2.15rem !important;
    font-weight: 800 !important;
    line-height: 1.15 !important;
    letter-spacing: -0.03em !important;
    color: #111827 !important;
    margin: 0 0 10px 0 !important;
}

@media (min-width: 768px) {
    .card-grey-headline {
        font-size: 2.55rem !important;
    }
}

.card-grey-sub {
    font-size: 0.92rem !important;
    line-height: 1.5 !important;
    color: #4B5563 !important;
    margin: 0 !important;
    max-width: 480px !important;
}

/* Primary Pill Button */
.btn-pill-primary {
    background-color: #0A0A0A !important;
    color: #FFFFFF !important;
    border-radius: 9999px !important;
    padding: 12px 28px !important;
    font-size: 0.98rem !important;
    font-weight: 600 !important;
    border: none !important;
    cursor: pointer !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    transition: transform 0.15s ease, background-color 0.15s ease !important;
}
.btn-pill-primary:hover {
    background-color: #262626 !important;
    transform: translateY(-1px) !important;
}

.btn-pill-outline {
    background-color: transparent !important;
    color: #111827 !important;
    border: 1.5px solid #111827 !important;
    border-radius: 9999px !important;
    padding: 12px 26px !important;
    font-size: 0.98rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
.btn-pill-outline:hover {
    background-color: #111827 !important;
    color: #FFFFFF !important;
}

/* === Results and Benchmark Cards (White rounded cards, radius ~28px) === */
.bento-card-white {
    background-color: #FFFFFF !important;
    border-radius: 28px !important;
    padding: 26px 28px !important;
    margin-top: 24px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02) !important;
    border: 1px solid rgba(0, 0, 0, 0.05) !important;
}

/* Clean Input styling */
.gradio-container textarea, 
.gradio-container input[type="text"],
.gradio-container input[type="number"] {
    border-radius: 16px !important;
    border: 1.5px solid #E2E8F0 !important;
    background-color: #FFFFFF !important;
    color: #111827 !important;
    font-size: 0.92rem !important;
    padding: 12px 14px !important;
    transition: border-color 0.2s ease !important;
}

.gradio-container textarea:focus, 
.gradio-container input[type="text"]:focus {
    border-color: #5B9FFF !important;
    box-shadow: 0 0 0 3px rgba(91, 159, 255, 0.15) !important;
    outline: none !important;
}

/* Tables in Results */
.gradio-container table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid #E2E8F0 !important;
    width: 100% !important;
}
.gradio-container th {
    background-color: #F8FAFC !important;
    color: #334155 !important;
    font-weight: 700 !important;
    font-size: 0.84rem !important;
    padding: 12px 14px !important;
    border-bottom: 1px solid #E2E8F0 !important;
}
.gradio-container td {
    padding: 12px 14px !important;
    border-bottom: 1px solid #F1F5F9 !important;
    font-size: 0.88rem !important;
    color: #1E293B !important;
}
.gradio-container tr:last-child td {
    border-bottom: none !important;
}

/* File downloads */
.gradio-container .gr-file {
    border-radius: 18px !important;
    border: 1px dashed #CBD5E1 !important;
    background: #F8FAFC !important;
}
"""
