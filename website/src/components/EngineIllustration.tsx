import { useId } from 'react'

export default function EngineIllustration() {
  const id = useId().replace(/:/g, '')
  const fill = (name: string) => `url(#${id}-${name})`

  return <svg viewBox="0 0 700 440" role="img" aria-label="Animated conceptual cutaway of a turbofan engine. Decorative motion, not live engine telemetry." className="engine-svg">
    <defs>
      <linearGradient id={`${id}-shell`} x1="0" x2="0" y1="0" y2="1">
        <stop stopColor="#d3e2e9"/><stop offset=".13" stopColor="#728594"/><stop offset=".35" stopColor="#34424e"/><stop offset=".73" stopColor="#101820"/><stop offset="1" stopColor="#687f91"/>
      </linearGradient>
      <linearGradient id={`${id}-rim`} x1="0" x2="1" y1="0" y2="1">
        <stop stopColor="#edf5f7"/><stop offset=".25" stopColor="#97afbf"/><stop offset=".5" stopColor="#354858"/><stop offset=".76" stopColor="#aac3d0"/><stop offset="1" stopColor="#45617a"/>
      </linearGradient>
      <radialGradient id={`${id}-fan`}>
        <stop stopColor="#1b2935"/><stop offset=".7" stopColor="#101a24"/><stop offset="1" stopColor="#344a5a"/>
      </radialGradient>
      <linearGradient id={`${id}-blade`} x1="0" x2="1" y1="0" y2="1">
        <stop stopColor="#a5bdcb"/><stop offset=".45" stopColor="#597789"/><stop offset="1" stopColor="#172735"/>
      </linearGradient>
      <linearGradient id={`${id}-core`}>
        <stop stopColor="#162b3b"/><stop offset=".45" stopColor="#53849d"/><stop offset=".75" stopColor="#1d3a4b"/><stop offset="1" stopColor="#131e29"/>
      </linearGradient>
      <linearGradient id={`${id}-flow`}>
        <stop stopColor="#8cbce6" stopOpacity="0"/><stop offset=".45" stopColor="#8cbce6" stopOpacity=".65"/><stop offset="1" stopColor="#8cbce6" stopOpacity="0"/>
      </linearGradient>
    </defs>

    <g className="engine-orbits" fill="none" stroke="#90b2ce">
      <ellipse cx="348" cy="228" rx="294" ry="181" strokeOpacity=".09"/>
      <ellipse cx="348" cy="228" rx="250" ry="143" strokeOpacity=".08" strokeDasharray="3 10"/>
      <path d="M47 229H663M347 31V414" strokeOpacity=".09" strokeDasharray="2 8"/>
    </g>

    <g className="engine-assembly">
      <g transform="translate(48 43) rotate(-9 315 175)">
        <ellipse cx="318" cy="319" rx="241" ry="18" fill="#010306" opacity=".45"/>
        <path d="M132 76L449 111Q482 118 532 150L566 164V204L532 219Q485 248 449 256L132 290Z" fill={fill('shell')} stroke="#9bb7ca" strokeOpacity=".65" strokeWidth="1.1"/>
        <path d="M174 107L440 130L510 164V205L440 235L174 260Z" fill="#101b24" stroke="#698a9e" strokeOpacity=".65"/>
        <path d="M189 115L438 137L484 159L438 161L189 147Z" fill="#718997" opacity=".22"/>
        <path d="M182 257L442 230L508 204" fill="none" stroke="#d0dee5" strokeOpacity=".55"/>
        <path d="M193 171L435 172L534 184L539 193L435 208L193 217Z" fill={fill('core')} stroke="#7bb3cd" strokeWidth="1.1"/>

        <g stroke="#8eaebe" strokeOpacity=".68" fill="none">
          {[222, 251, 280, 309].map((x, i) => <g key={x}>
            <path d={`M${x} ${135 + i * 2}l19 36v44l-19 ${31 - i * 3}`} strokeWidth="1.5"/>
            <path d={`M${x + 7} ${141 + i * 2}l18 30v44l-18 ${24 - i * 3}`} strokeOpacity=".32"/>
          </g>)}
          <path d="M354 146l14 25v42l-14 24M388 150l14 23v37l-14 24M421 153l12 20v36l-12 22" strokeWidth="2"/>
        </g>
        <path d="M327 169L375 171L375 210L327 214Z" fill="#b9d7e2" opacity=".12" stroke="#a2d7e2"/>
        <path d="M530 154l34 10l14 13v28l-14 11l-34 6" fill="#263a48" stroke="#8ba9bc"/>
        <path d="M549 168l17 6v31l-17 7" fill="none" stroke="#a1bfca" strokeOpacity=".7"/>
        <path d="M212 86V64H321V96M237 278v21h106v-33" fill="none" stroke="#708b9b" strokeWidth="2"/>

        <ellipse cx="141" cy="184" rx="83" ry="118" fill={fill('rim')} stroke="#c6d8df" strokeWidth="1.2"/>
        <ellipse cx="136" cy="184" rx="73" ry="106" fill="#0c151f" stroke="#708a9c" strokeWidth="1.3"/>
        <ellipse cx="136" cy="184" rx="65" ry="97" fill={fill('fan')} stroke="#9ab7c7" strokeOpacity=".5"/>

        <g transform="translate(136 184) scale(.68 1)">
          <g className="engine-fan" fill={fill('blade')} stroke="#aac6d4" strokeOpacity=".58" strokeWidth=".7">
            {Array.from({ length: 22 }, (_, i) => <path key={i} d="M-4 -19Q17 -39 6 -93Q-7 -92 -18 -81Q-3 -55 -13 -23Z" transform={`rotate(${i * (360 / 22)})`}/>)}
          </g>
        </g>
        <ellipse cx="136" cy="184" rx="16" ry="23" fill={fill('rim')} stroke="#c4d6de"/>
        <ellipse cx="132" cy="180" rx="6" ry="10" fill="#c7dce7" opacity=".45"/>

        <g className="engine-flow" fill="none" stroke={fill('flow')} strokeWidth="1.2" strokeDasharray="35 180">
          <path d="M24 113C191 58 396 81 615 155"/>
          <path d="M18 291C182 340 399 313 612 221"/>
        </g>
        <g fill="#a8d5f0">
          <circle className="engine-sensor" cx="254" cy="95" r="3"/>
          <circle className="engine-sensor" cx="403" cy="235" r="3"/>
          <circle className="engine-sensor" cx="546" cy="157" r="3"/>
        </g>
        <g stroke="#89abc3" strokeOpacity=".6" strokeWidth=".8" fill="none">
          <path d="M254 95V31H302"/><path d="M403 235V324H450"/>
        </g>
        <g fill="#8ba6b9" fontFamily="'IBM Plex Mono', monospace" fontSize="9" letterSpacing="1.2">
          <text x="308" y="34">COMPRESSOR</text><text x="457" y="327">TURBINE CORE</text>
        </g>
      </g>
    </g>
    <text x="62" y="412" fill="#688093" fontFamily="'IBM Plex Mono', monospace" fontSize="8" letterSpacing="1.6">CONCEPTUAL CUTAWAY · NOT TO SCALE</text>
  </svg>
}
