import * as THREE from 'three'
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js'

export interface EngineSceneController {
  setMotion: (enabled: boolean) => void
  dispose: () => void
}

/** A procedural illustrative turbofan. Geometry and rotation are not measured telemetry. */
export function createEngineScene(container: HTMLDivElement, initialMotion: boolean): EngineSceneController {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'low-power' })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.6))
  renderer.setClearColor(0x000000, 0)
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 1.1
  renderer.domElement.setAttribute('aria-hidden', 'true')
  container.appendChild(renderer.domElement)

  const scene = new THREE.Scene()
  const camera = new THREE.PerspectiveCamera(36, 1, .1, 80)
  camera.position.set(6.6, 3.1, 10.1)
  camera.lookAt(0, 0, -.15)

  const pmrem = new THREE.PMREMGenerator(renderer)
  const room = new RoomEnvironment()
  const environment = pmrem.fromScene(room, .04)
  scene.environment = environment.texture
  pmrem.dispose()
  room.dispose()

  const key = new THREE.DirectionalLight(0xdcecff, 5)
  key.position.set(-4, 7, 5)
  scene.add(key)
  const blue = new THREE.DirectionalLight(0x527bff, 5)
  blue.position.set(6, 1, -4)
  scene.add(blue)
  const edge = new THREE.DirectionalLight(0xffffff, 3)
  edge.position.set(-5, -3, 3)
  scene.add(edge, new THREE.AmbientLight(0xb8d5f5, .3))

  const engine = new THREE.Group()
  engine.rotation.z = -.12
  scene.add(engine)
  const shell = new THREE.MeshStandardMaterial({ color: 0x384350, metalness: .92, roughness: .3, side: THREE.DoubleSide })
  const silver = new THREE.MeshStandardMaterial({ color: 0xb1bac5, metalness: .94, roughness: .24 })
  const dark = new THREE.MeshStandardMaterial({ color: 0x0e1721, metalness: .75, roughness: .4, side: THREE.DoubleSide })
  const bladeMaterial = new THREE.MeshStandardMaterial({ color: 0x687b90, metalness: .9, roughness: .3, side: THREE.DoubleSide })
  const accent = new THREE.MeshStandardMaterial({ color: 0x588acb, metalness: .6, roughness: .35, emissive: 0x17304e, emissiveIntensity: .3 })

  const profile = [
    new THREE.Vector2(1.35, -2.8), new THREE.Vector2(1.44, -2.55),
    new THREE.Vector2(1.64, -1.7), new THREE.Vector2(1.97, -.7),
    new THREE.Vector2(2.12, .8), new THREE.Vector2(2.14, 1.72),
  ]
  const housing = new THREE.Mesh(new THREE.LatheGeometry(profile, 100), shell)
  housing.rotation.x = Math.PI / 2
  engine.add(housing)

  const ring = (radius: number, tube: number, z: number, material: THREE.Material) => {
    const mesh = new THREE.Mesh(new THREE.TorusGeometry(radius, tube, 12, 100), material)
    mesh.position.z = z
    engine.add(mesh)
    return mesh
  }
  ring(2.12, .12, 1.76, silver)
  ring(1.98, .045, 1.81, dark)
  ring(2.12, .018, 1.37, accent)
  ring(2.04, .024, .2, dark)
  ring(1.91, .025, -.87, silver)
  ring(1.66, .02, -1.65, dark)
  ring(1.39, .07, -2.72, silver)

  const innerHousing = new THREE.Mesh(new THREE.CylinderGeometry(1.96, 1.7, 1.1, 80, 1, true), dark)
  innerHousing.rotation.x = Math.PI / 2
  innerHousing.position.z = 1.15
  engine.add(innerHousing)

  const backplate = new THREE.Mesh(new THREE.CircleGeometry(1.82, 80), dark)
  backplate.position.z = .9
  engine.add(backplate)
  const rotor = new THREE.Group()
  rotor.position.z = 1.48
  engine.add(rotor)

  // Swept, twisted airfoil surfaces give the fan genuine depth and separate blades.
  const positions: number[] = [], indices: number[] = []
  const radialSteps = 25, chordSteps = 7
  for (let radial = 0; radial <= radialSteps; radial++) {
    const t = radial / radialSteps
    const radius = .34 + t * 1.54
    const sweep = .39 * t * t - .08 * t
    const chordAngle = .31 - t * .17
    for (let chord = 0; chord <= chordSteps; chord++) {
      const u = chord / chordSteps
      const angle = sweep + (u - .5) * chordAngle
      const depth = Math.sin(u * Math.PI) * .09 + (u - .5) * (.46 - t * .22) + t * .11
      positions.push(Math.cos(angle) * radius, Math.sin(angle) * radius, depth)
    }
  }
  for (let radial = 0; radial < radialSteps; radial++) {
    for (let chord = 0; chord < chordSteps; chord++) {
      const a = radial * (chordSteps + 1) + chord, b = a + chordSteps + 1
      indices.push(a, b, a + 1, b, b + 1, a + 1)
    }
  }
  const bladeGeometry = new THREE.BufferGeometry()
  bladeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  bladeGeometry.setIndex(indices)
  bladeGeometry.computeVertexNormals()
  for (let i = 0; i < 28; i++) {
    const mesh = new THREE.Mesh(bladeGeometry, bladeMaterial)
    mesh.rotation.z = i * Math.PI * 2 / 28
    rotor.add(mesh)
  }

  const spinnerProfile = [new THREE.Vector2(.34, 0), new THREE.Vector2(.31, .16), new THREE.Vector2(.24, .36), new THREE.Vector2(.12, .62), new THREE.Vector2(.045, .74), new THREE.Vector2(0, .77)]
  const spinner = new THREE.Mesh(new THREE.LatheGeometry(spinnerProfile, 64), silver)
  spinner.rotation.x = Math.PI / 2
  spinner.position.z = 1.57
  engine.add(spinner)
  ring(.36, .035, 1.58, silver)

  // Nacelle fasteners and longitudinal seams add scale without invented instrument values.
  const boltGeometry = new THREE.SphereGeometry(.035, 8, 6)
  for (let i = 0; i < 24; i++) {
    const angle = i * Math.PI * 2 / 24
    const bolt = new THREE.Mesh(boltGeometry, silver)
    bolt.position.set(Math.cos(angle) * 2.12, Math.sin(angle) * 2.12, 1.9)
    engine.add(bolt)
  }
  const seams = new THREE.Group()
  const seamMaterial = new THREE.LineBasicMaterial({ color: 0x7d97af, transparent: true, opacity: .25 })
  for (let i = 0; i < 12; i++) {
    const angle = i * Math.PI * 2 / 12
    const points = profile.map(point => new THREE.Vector3(Math.cos(angle) * (point.x + .005), Math.sin(angle) * (point.x + .005), point.y))
    seams.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), seamMaterial))
  }
  engine.add(seams)

  let motion = initialMotion
  let visible = true
  let disposed = false
  let frame = 0
  let lastTime = 0
  const pointer = { x: 0, y: 0 }
  const render = () => { if (!disposed) renderer.render(scene, camera) }
  const tick = (time: number) => {
    frame = 0
    if (disposed || !motion || !visible || document.hidden) return
    const delta = Math.min((time - (lastTime || time)) / 1000, .05)
    lastTime = time
    rotor.rotation.z += delta * .13
    engine.rotation.y += (pointer.x * .12 - engine.rotation.y) * .035
    engine.rotation.x += (pointer.y * .08 - engine.rotation.x) * .035
    engine.position.y = Math.sin(time * .00045) * .035
    render()
    frame = requestAnimationFrame(tick)
  }
  const start = () => {
    if (frame || disposed || !motion || !visible || document.hidden) return
    lastTime = 0
    frame = requestAnimationFrame(tick)
  }
  const stop = () => { cancelAnimationFrame(frame); frame = 0 }
  const resize = () => {
    const width = container.clientWidth, height = container.clientHeight
    if (!width || !height) return
    renderer.setSize(width, height)
    camera.aspect = width / height
    camera.updateProjectionMatrix()
    render()
  }
  const resizeObserver = new ResizeObserver(resize)
  resizeObserver.observe(container)
  const visibilityObserver = new IntersectionObserver(entries => {
    visible = entries[0].isIntersecting
    visible ? start() : stop()
  })
  visibilityObserver.observe(container)
  const move = (event: PointerEvent) => {
    if (!motion || event.pointerType === 'touch') return
    const rect = container.getBoundingClientRect()
    pointer.x = ((event.clientX - rect.left) / rect.width - .5) * 2
    pointer.y = ((event.clientY - rect.top) / rect.height - .5) * 2
  }
  const leave = () => { pointer.x = 0; pointer.y = 0 }
  const visibilityChange = () => document.hidden ? stop() : start()
  container.addEventListener('pointermove', move)
  container.addEventListener('pointerleave', leave)
  document.addEventListener('visibilitychange', visibilityChange)
  resize()
  start()

  return {
    setMotion(enabled) {
      motion = enabled
      if (enabled) start()
      else { stop(); render() }
    },
    dispose() {
      disposed = true
      stop()
      resizeObserver.disconnect()
      visibilityObserver.disconnect()
      container.removeEventListener('pointermove', move)
      container.removeEventListener('pointerleave', leave)
      document.removeEventListener('visibilitychange', visibilityChange)
      const geometries = new Set<THREE.BufferGeometry>()
      const materials = new Set<THREE.Material>()
      engine.traverse(object => {
        if (object instanceof THREE.Mesh || object instanceof THREE.Line) {
          geometries.add(object.geometry)
          const values = Array.isArray(object.material) ? object.material : [object.material]
          values.forEach(material => materials.add(material))
        }
      })
      geometries.forEach(geometry => geometry.dispose())
      materials.forEach(material => material.dispose())
      environment.dispose()
      renderer.dispose()
      renderer.forceContextLoss()
      renderer.domElement.remove()
    },
  }
}
