export default function EngineIllustration() {
  return <svg viewBox="0 0 650 320" role="img" aria-label="Conceptual technical illustration of a turbofan engine" className="engine-svg">
    <defs>
      <linearGradient id="shell" x1="0" x2="0" y1="0" y2="1"><stop stopColor="#496275"/><stop offset=".25" stopColor="#172a39"/><stop offset=".66" stopColor="#0a1926"/><stop offset="1" stopColor="#31485b"/></linearGradient>
      <linearGradient id="core" x1="0" x2="1"><stop stopColor="#152c3d"/><stop offset=".55" stopColor="#233e4f"/><stop offset="1" stopColor="#08151f"/></linearGradient>
      <radialGradient id="fan"><stop stopColor="#122536"/><stop offset=".62" stopColor="#0a1723"/><stop offset=".85" stopColor="#37566b"/><stop offset="1" stopColor="#081620"/></radialGradient>
      <pattern id="dots" width="14" height="14" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".65" fill="#4a819a" opacity=".27"/></pattern>
    </defs>
    <rect width="650" height="320" fill="url(#dots)" opacity=".55"/>
    <g stroke="#54788c" strokeWidth="1" opacity=".35"><path d="M20 160H635M327 28v264" strokeDasharray="3 8"/><path d="M30 42h40M30 42v24M616 42h-40M616 42v24M30 278h40M30 278v-24M616 278h-40M616 278v-24"/></g>
    <g transform="translate(70 17)">
      <path d="M100 87 L419 99 Q460 114 499 135 L521 142 L521 160 L499 169 Q457 188 419 203 L100 215 Z" fill="url(#shell)" stroke="#6a91a4" strokeWidth="1.2"/>
      <path d="M155 113 L422 121 L476 143 L476 160 L422 180 L155 190Z" fill="#081b28" stroke="#456b7e"/>
      <path d="M188 135 L425 137 L485 148 L488 154 L425 166 L188 171Z" fill="url(#core)" stroke="#6998ac" strokeWidth="1"/>
      <path d="M216 113l28 22v34l-28 21M260 114l25 21v34l-25 19M304 116l20 20v31l-20 20M346 117l17 20v29l-17 19M389 119l15 19v28l-15 17" fill="none" stroke="#6f91a4" opacity=".65"/>
      <path d="M119 93l28 20v85l-28 14" fill="none" stroke="#8db5c5" strokeWidth="2" opacity=".6"/>
      <ellipse cx="108" cy="151" rx="73" ry="113" fill="#0a1a27" stroke="#7297a9" strokeWidth="2"/>
      <ellipse cx="104" cy="151" rx="61" ry="99" fill="url(#fan)" stroke="#6093a9" strokeWidth="1.4"/>
      <g transform="translate(104 151)" fill="#355a6d" stroke="#7aa6b6" strokeWidth=".8">
        {Array.from({length: 16}, (_,i) => <path key={i} d="M0 -12 Q23 -35 11 -84 Q-6 -77 -9 -18Z" transform={`rotate(${i*22.5})`}/>)}
      </g>
      <ellipse cx="104" cy="151" rx="18" ry="25" fill="#0c1a25" stroke="#94bcc7" strokeWidth="1.4"/><ellipse cx="104" cy="151" rx="8" ry="12" fill="#739dac" opacity=".65"/>
      <path d="M59 70Q108 39 150 67M59 233Q108 261 150 235" fill="none" stroke="#7aa9bc" opacity=".6"/>
      <path d="M507 136h21l9 10v13l-9 11h-21" fill="#172d3c" stroke="#6692a5"/>
      <path d="M220 90v-18h99v18M223 211v17h101v-17" fill="none" stroke="#60899b" opacity=".5"/>
      <circle cx="216" cy="85" r="3" fill="#56b6ca"/><circle cx="353" cy="185" r="3" fill="#56b6ca"/><circle cx="476" cy="145" r="3" fill="#56b6ca"/>
      <g stroke="#75b9cc" opacity=".5" strokeWidth=".8" fill="none"><path d="M216 85v-34h45"/><path d="M353 185v70h45"/><path d="M476 145v-70h38"/></g>
      <g fill="#94b7c4" fontFamily="monospace" fontSize="9" letterSpacing="1"><text x="264" y="54">FAN INLET</text><text x="400" y="258">CORE ASSEMBLY</text><text x="517" y="78">EXHAUST</text></g>
    </g>
    <g fill="#5f8a9b" fontFamily="monospace" fontSize="9" letterSpacing="1.5"><text x="19" y="306">TURBOFAN / SCHEMATIC VIEW</text><text x="507" y="306">NOT TO SCALE</text></g>
  </svg>
}
