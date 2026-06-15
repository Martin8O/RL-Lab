// Per-environment "how to play" guides for the Play-vs-AI instructions popup (E2).
// Data-driven and bilingual, mirroring content/parameters.ts: adding a game is a content-only
// change. Static chrome (button label, section titles) lives in i18n; the per-env prose here.

import type { Bilingual } from '../api/types'

export interface PlayControl {
  /** The key(s) for this action, shown verbatim (e.g. "← / A"). */
  keys: string
  action: Bilingual
}

export interface PlayGuide {
  goal: Bilingual
  controls: PlayControl[]
  tips: Bilingual
}

// MiniGrid (G2c) — shared controls + tips across the family (the per-env goal differs). The agent has
// a facing direction (turn, then move forward), play is turn-based, and the reward is sparse.
const MINIGRID_CONTROLS: PlayControl[] = [
  { keys: '← / A', action: { en: 'Turn left', cz: 'Otočit se doleva' } },
  { keys: '→ / D', action: { en: 'Turn right', cz: 'Otočit se doprava' } },
  { keys: '↑ / W', action: { en: 'Move forward', cz: 'Krok vpřed' } },
  { keys: 'P / Enter', action: { en: 'Pick up (key / ball)', cz: 'Sebrat (klíč / míček)' } },
  { keys: 'O', action: { en: 'Drop the carried key', cz: 'Odložit nesený klíč' } },
  { keys: 'Space / E', action: { en: 'Toggle — open a door', cz: 'Interakce — otevřít dveře' } },
]

const MINIGRID_TIPS: Bilingual = {
  en: 'You do not slide around the grid like Toy Text — you have a facing direction, so turn (← / →) to '
    + 'face the way you want to go, then step forward (↑). Picking up (P), dropping (O) and opening a '
    + 'door (Space) all act on the square directly in front of you, and you can carry only one object '
    + 'at a time. It is turn-based: one move per key press, so '
    + 'take your time. The reward is sparse — you score nothing until you reach the goal (or pick up the '
    + 'ball), so be patient and explore. The small 7×7 window around the agent is all it actually sees. '
    + 'Switch to "Watch AI" to see a trained policy explore efficiently.',
  cz: 'Nekloužete po mřížce jako u Toy Textu — máte směr, kterým koukáte, takže se nejdřív otočte (← / →) '
    + 'tam, kam chcete jít, a pak udělejte krok vpřed (↑). Sebrání (P), odložení (O) i otevření dveří '
    + '(mezerník) působí na políčko přímo před vámi a najednou unesete jen jednu věc. Hra je tahová: '
    + 'jeden pohyb na stisk klávesy, '
    + 'takže nespěchejte. Odměna je řídká — dokud nedojdete do cíle (nebo neseberete míček), nezískáte nic, '
    + 'takže buďte trpěliví a zkoumejte. Malé okno 7×7 kolem agenta je vše, co sám vidí. Přepněte na '
    + '„Sledovat AI“ a uvidíte, jak natrénovaná strategie zkoumá efektivně.',
}

// BipedalWalker (+ Hardcore) (G3b) — shared controls + tips (only the per-env goal differs). The
// action is continuous Box(4): four leg-joint torques, two per leg. Arrows drive leg 1, WASD leg 2;
// each pair pushes one joint each way, and holding several keys moves several joints at once.
const BIPEDAL_CONTROLS: PlayControl[] = [
  { keys: '← / →', action: { en: 'Leg 1 hip — torque each way', cz: 'Noha 1, kyčel — moment každým směrem' } },
  { keys: '↑ / ↓', action: { en: 'Leg 1 knee — torque each way', cz: 'Noha 1, koleno — moment každým směrem' } },
  { keys: 'A / D', action: { en: 'Leg 2 hip — torque each way', cz: 'Noha 2, kyčel — moment každým směrem' } },
  { keys: 'W / S', action: { en: 'Leg 2 knee — torque each way', cz: 'Noha 2, koleno — moment každým směrem' } },
  { keys: '(release)', action: { en: 'No torque — the legs go limp', cz: 'Bez momentu — nohy ochabnou' } },
]

const BIPEDAL_TIPS: Bilingual = {
  en: 'You steer all four leg joints at once, so this is genuinely hard by hand — which is exactly why '
    + 'an AI is trained for it. Try to find a marching rhythm: push one leg\'s hip forward while bending '
    + 'the other leg\'s knee, then swap, and keep making small corrections so the body stays upright (tip '
    + 'too far and you fall). Hold several keys together to move more than one joint at the same time, and '
    + 'lower the play speed (down to 0.1×) to give yourself time to react. Switch to "Watch AI" once a '
    + 'model has been trained for this on a GPU.',
  cz: 'Řídíte všechny čtyři klouby nohou najednou, takže je to rukama vážně těžké — přesně proto se na to '
    + 'cvičí AI. Zkuste najít pochodový rytmus: přitlačte kyčel jedné nohy vpřed a zároveň pokrčte koleno '
    + 'druhé nohy, pak je prohoďte a stále dělejte malé korekce, ať tělo zůstane vzpřímené (když se nakloní '
    + 'moc, upadnete). Podržením více kláves najednou pohnete více klouby současně a snížením rychlosti '
    + 'hraní (až na 0,1×) získáte čas reagovat. Až bude na to natrénovaný model na GPU, přepněte na '
    + '„Sledovat AI“.',
}

export const PLAY_GUIDES: Record<string, PlayGuide> = {
  cartpole: {
    goal: {
      en: 'Keep the pole balanced upright for as long as you can. Every step you survive is +1 '
        + 'point — reach 500 to fully solve it. The episode ends the moment the pole tips too far '
        + 'or the cart runs off either edge.',
      cz: 'Udržte tyč co nejdéle ve svislé poloze. Každý krok, který přežijete, je +1 bod — '
        + '500 bodů znamená plné vyřešení. Epizoda končí ve chvíli, kdy se tyč příliš nakloní nebo '
        + 'vozík vyjede z některého okraje.',
    },
    controls: [
      { keys: '← / A', action: { en: 'Push the cart left', cz: 'Tlačit vozík doleva' } },
      { keys: '→ / D', action: { en: 'Push the cart right', cz: 'Tlačit vozík doprava' } },
    ],
    tips: {
      en: "There is no \"do nothing\" action — the cart always moves, so make small, frequent "
        + 'corrections rather than big swings. Switch to "Watch AI" to see how a trained agent '
        + 'keeps it upright.',
      cz: 'Neexistuje volba „nedělat nic“ — vozík se hýbe pořád, takže korigujte malými a častými '
        + 'pohyby místo velkých výkyvů. Přepněte na „Sledovat AI“ a uvidíte, jak tyč udrží '
        + 'natrénovaný agent.',
    },
  },

  lunarlander: {
    goal: {
      en: 'Land the module gently on the pad between the two flags, upright and slow, with both '
        + 'legs down. A soft, on-target landing scores around +200 (the "solved" mark); crashing '
        + 'or flying off the screen ends the episode with a big penalty. Wasting fuel costs a little '
        + 'each frame, so thrust only when you need to.',
      cz: 'Přistaňte s modulem jemně na plošinu mezi dvěma vlajkami — narovnaný, pomalu a s oběma '
        + 'nohama dole. Měkké přistání na cíli dá kolem +200 (hranice „vyřešeno“); havárie nebo '
        + 'vylétnutí z obrazovky ukončí epizodu velkou penalizací. Plýtvání palivem stojí každý '
        + 'snímek trochu, takže přidávejte tah jen když je potřeba.',
    },
    controls: [
      { keys: '↑ / W', action: { en: 'Fire the main engine (thrust up)', cz: 'Zážeh hlavního motoru (tah vzhůru)' } },
      { keys: '← / A', action: { en: 'Fire the left thruster (nudge left)', cz: 'Zážeh levé trysky (posun doleva)' } },
      { keys: '→ / D', action: { en: 'Fire the right thruster (nudge right)', cz: 'Zážeh pravé trysky (posun doprava)' } },
      { keys: '(release)', action: { en: 'No thrust — coast and fall', cz: 'Bez tahu — volný pád' } },
    ],
    tips: {
      en: 'Releasing every key cuts all engines, so let gravity do the work and tap the thrusters '
        + 'in short bursts. Kill your sideways drift early, then feather the main engine to slow your '
        + 'descent just before touchdown. Switch to "Watch AI" to see how a trained agent lands it.',
      cz: 'Puštění všech kláves vypne motory, takže nechte pracovat gravitaci a trysky používejte '
        + 'v krátkých dávkách. Boční pohyb zastavte včas a těsně před dosednutím zpomalujte sestup '
        + 'jemným přidáváním hlavního motoru. Přepněte na „Sledovat AI“ a uvidíte, jak přistává '
        + 'natrénovaný agent.',
    },
  },

  mountaincar: {
    goal: {
      en: 'Reach the flag on top of the right-hand hill. The engine is too weak to drive straight '
        + 'up, so you must build momentum by rocking back and forth. Every step costs −1 point, so '
        + 'the quicker you get there the higher (less negative) your score — reaching the flag scores '
        + 'around −110, the "solved" mark. The episode ends at the flag or after 200 steps.',
      cz: 'Dojeďte k vlajce na vrcholu pravého kopce. Motor je příliš slabý na přímý výjezd, takže '
        + 'musíte nabrat setrvačnost houpáním sem a tam. Každý krok stojí −1 bod, takže čím dřív se '
        + 'tam dostanete, tím vyšší (méně záporné) skóre — dosažení vlajky dá kolem −110, hranice '
        + '„vyřešeno“. Epizoda končí u vlajky nebo po 200 krocích.',
    },
    controls: [
      { keys: '← / A', action: { en: 'Accelerate left', cz: 'Zrychlit doleva' } },
      { keys: '→ / D', action: { en: 'Accelerate right', cz: 'Zrychlit doprava' } },
      { keys: '(release)', action: { en: 'No throttle — coast', cz: 'Bez plynu — setrvačnost' } },
    ],
    tips: {
      en: 'You cannot power straight up the right hill. First drive left up the back slope to gain '
        + 'height, then let gravity pull you down and across — and add throttle in the direction you '
        + 'are already moving to swing higher each pass. Switch to "Watch AI" to see the momentum trick.',
      cz: 'Na pravý kopec se přímo nevyškrábete. Nejdřív vyjeďte doleva na zadní svah, naberte výšku '
        + 'a nechte se gravitací stáhnout dolů a přes — a přidávejte plyn ve směru, kterým už jedete, '
        + 'ať se každým průjezdem rozhoupete výš. Přepněte na „Sledovat AI“ a uvidíte ten trik se '
        + 'setrvačností.',
    },
  },

  acrobot: {
    goal: {
      en: 'Swing the free tip of the two-link arm up until it rises above the bar. You can only add '
        + 'torque at the middle joint, so you must pump like a child on a swing to build height. Every '
        + 'step costs −1 point, so swing up as fast as you can — a good run reaches the target around '
        + '−100, the "solved" mark. The episode ends on success or after 500 steps.',
      cz: 'Rozhoupejte volný konec dvoukloubového ramene tak, aby vystoupal nad tyč. Točivý moment '
        + 'lze přidat jen v prostředním kloubu, takže musíte „pumpovat“ jako dítě na houpačce, abyste '
        + 'nabrali výšku. Každý krok stojí −1 bod, takže se vyhoupněte co nejrychleji — dobrý běh '
        + 'dosáhne cíle kolem −100, hranice „vyřešeno“. Epizoda končí úspěchem nebo po 500 krocích.',
    },
    controls: [
      { keys: '← / A', action: { en: 'Apply torque one way', cz: 'Přidat moment jedním směrem' } },
      { keys: '→ / D', action: { en: 'Apply torque the other way', cz: 'Přidat moment druhým směrem' } },
      { keys: '(release)', action: { en: 'No torque — let it swing', cz: 'Bez momentu — nech houpat' } },
    ],
    tips: {
      en: 'Do not just push one way — time your torque with the arm\'s natural swing so you add energy '
        + 'on each pass (exactly like pumping a swing). Alternate left and right as it swings back and '
        + 'forth until the tip flips up over the bar. Switch to "Watch AI" to see the rhythm.',
      cz: 'Netlačte jen jedním směrem — načasujte moment podle přirozeného houpání ramene, ať při '
        + 'každém průkmitu přidáte energii (přesně jako při rozhoupávání houpačky). Střídejte levou a '
        + 'pravou, jak se kýve sem a tam, dokud se konec nepřehoupne přes tyč. Přepněte na „Sledovat '
        + 'AI“ a uvidíte ten rytmus.',
    },
  },

  pendulum: {
    goal: {
      en: 'Swing the pendulum up and balance it pointing straight up, as still as you can. This is the '
        + 'first continuous-control game: instead of buttons, the torque is a dial. Every step costs a '
        + 'little — more when the pendulum is far from upright, spinning fast, or you push hard — so the '
        + 'score is always negative and you want it as close to 0 as possible (a good run reaches around '
        + '−150). The episode always runs 200 steps; there is no crash, only the running cost.',
      cz: 'Vyhoupněte kyvadlo nahoru a udržte ho co nejklidněji ve svislé poloze špičkou vzhůru. Tohle je '
        + 'první hra se spojitým řízením: místo tlačítek je točivý moment plynulý „knoflík“. Každý krok '
        + 'něco stojí — víc, když je kyvadlo daleko od svislé polohy, rychle se točí nebo silně tlačíte — '
        + 'takže skóre je vždy záporné a chcete ho mít co nejblíž 0 (dobrý běh se dostane kolem −150). '
        + 'Epizoda trvá vždy 200 kroků; není žádná havárie, jen průběžná penalizace.',
    },
    controls: [
      { keys: '← / A', action: { en: 'Full torque counter-clockwise', cz: 'Plný moment proti směru hodin' } },
      { keys: '→ / D', action: { en: 'Full torque clockwise', cz: 'Plný moment po směru hodin' } },
      { keys: '(release)', action: { en: 'No torque — let it swing', cz: 'Bez momentu — nech houpat' } },
    ],
    tips: {
      en: 'From hanging down you usually cannot push straight up — pump left and right in time with the '
        + 'swing to build height (like Acrobot), then, as it nears the top, ease off and make small '
        + 'corrections to hold it upright. Holding full torque wastes points once you are balanced. '
        + 'Switch to "Watch AI" to see a smooth swing-up and hold.',
      cz: 'Ze svěšené polohy se obvykle přímo nahoru nevytlačíte — pumpujte doleva a doprava do rytmu '
        + 'houpání, ať naberete výšku (jako u Acrobotu), a jak se blíží k vrcholu, uberte a malými '
        + 'korekcemi ho udržte svisle. Držet plný moment, když už balancujete, jen plýtvá body. '
        + 'Přepněte na „Sledovat AI“ a uvidíte plynulé vyhoupnutí a udržení.',
    },
  },

  mountaincarcontinuous: {
    goal: {
      en: 'Reach the flag on the right-hand hill — same hill as MountainCar, but the throttle is now '
        + 'continuous (you dial how hard to push, not just left/right/off). Reaching the flag pays a big '
        + '+100, while using force costs a tiny amount, so doing nothing scores 0 and a successful run '
        + 'scores around +90 (the "solved" mark). The episode ends at the flag or after 999 steps.',
      cz: 'Dojeďte k vlajce na pravém kopci — stejný kopec jako u MountainCar, ale plyn je teď spojitý '
        + '(nastavujete, jak silně tlačit, ne jen doleva/doprava/nic). Dosažení vlajky vyplatí velkých '
        + '+100, zatímco použití síly stojí drobnost, takže nicnedělání dá 0 a úspěšný běh kolem +90 '
        + '(hranice „vyřešeno“). Epizoda končí u vlajky nebo po 999 krocích.',
    },
    controls: [
      { keys: '← / A', action: { en: 'Full throttle left', cz: 'Plný plyn doleva' } },
      { keys: '→ / D', action: { en: 'Full throttle right', cz: 'Plný plyn doprava' } },
      { keys: '(release)', action: { en: 'No throttle — coast', cz: 'Bez plynu — setrvačnost' } },
    ],
    tips: {
      en: 'The engine is too weak to climb straight up, so use the same trick as MountainCar: drive left '
        + 'up the back slope, then throttle in the direction you are already moving to swing higher each '
        + 'pass until you crest the right hill. Because force costs points, do not pin the throttle '
        + 'pointlessly. Switch to "Watch AI" — or try Neuroevolution, whose population search is good at '
        + 'finding the flag.',
      cz: 'Motor je příliš slabý na přímý výjezd, takže použijte stejný trik jako u MountainCar: vyjeďte '
        + 'doleva na zadní svah a pak přidávejte plyn ve směru, kterým už jedete, ať se každým průjezdem '
        + 'rozhoupete výš, dokud nepřejedete pravý kopec. Protože síla stojí body, nedržte plyn zbytečně '
        + 'na doraz. Přepněte na „Sledovat AI“ — nebo zkuste Neuroevoluci, jejíž populační hledání vlajku '
        + 'nachází dobře.',
    },
  },

  bipedalwalker: {
    goal: {
      en: 'Walk the two-legged robot as far to the right as you can without falling over. You drive '
        + 'all four leg joints directly — a hip and a knee on each leg. Moving forward earns points, '
        + 'the motors cost a little, and a fall ends the run with a −100 penalty; a smooth walk to the '
        + 'far end scores about +300 (the "solved" mark). The episode ends on a fall or after 1600 steps.',
      cz: 'Doveďte dvounohého robota co nejdál doprava, aniž by upadl. Ovládáte přímo všechny čtyři '
        + 'klouby nohou — kyčel a koleno na každé noze. Pohyb vpřed dává body, motory něco stojí a pád '
        + 'ukončí běh penalizací −100; plynulá chůze až na konec dá kolem +300 (hranice „vyřešeno“). '
        + 'Epizoda končí pádem nebo po 1600 krocích.',
    },
    controls: BIPEDAL_CONTROLS,
    tips: BIPEDAL_TIPS,
  },

  bipedalwalkerhardcore: {
    goal: {
      en: 'The same walk to the right, but now over an obstacle course of ladders, stumps and pits — '
        + 'you have to climb, step around and leap across them. Same controls and scoring (points for '
        + 'progress, −100 for a fall, about +300 for finishing), just far harder. The episode ends on a '
        + 'fall or after 1600 steps.',
      cz: 'Stejná chůze doprava, ale teď přes překážkovou dráhu plnou žebříků, pařezů a jam — musíte je '
        + 'přelézt, obejít a přeskočit. Stejné ovládání i bodování (body za postup, −100 za pád, kolem '
        + '+300 za dojití), jen mnohem těžší. Epizoda končí pádem nebo po 1600 krocích.',
    },
    controls: BIPEDAL_CONTROLS,
    tips: BIPEDAL_TIPS,
  },

  // ── Toy Text grid-worlds (turn-based: one move per key press) ───────────────────────────────
  frozenlake: {
    goal: {
      en: 'Cross the frozen lake from the start (top-left) to the goal flag (bottom-right) without '
        + 'stepping into a hole (the dark circles). The ice is slippery: you only go the way you press '
        + 'about one time in three — the other two thirds the ice slides you to one side (never '
        + 'straight backwards). Reaching the goal scores 1; falling in a hole or running out of moves '
        + 'scores 0. (Want a version that always goes where you point? Pick "FrozenLake (4×4, no slip)".)',
      cz: 'Přejděte zamrzlé jezero ze startu (vlevo nahoře) k vlajce cíle (vpravo dole), aniž byste '
        + 'šlápli do díry (tmavé kruhy). Led klouže: tam, kam zmáčknete, půjdete jen asi jednou ze tří '
        + 'pokusů — ve zbylých dvou třetinách vás led smekne do strany (nikdy ne rovnou zpět). Dosažení '
        + 'cíle dá 1 bod; pád do díry nebo vyčerpání tahů dá 0. (Chcete verzi, kde jdete vždy tam, kam '
        + 'míříte? Zvolte „FrozenLake (4×4, bez kluzu)".)',
    },
    controls: [
      { keys: '← / A', action: { en: 'Move left', cz: 'Krok doleva' } },
      { keys: '→ / D', action: { en: 'Move right', cz: 'Krok doprava' } },
      { keys: '↑ / W', action: { en: 'Move up', cz: 'Krok nahoru' } },
      { keys: '↓ / S', action: { en: 'Move down', cz: 'Krok dolů' } },
    ],
    tips: {
      en: 'One move per key press (turn-based) — take your time. Because the ice slips, aim for paths '
        + 'that keep a margin from the holes rather than hugging their edges. Switch to "Watch AI" to '
        + 'see a learned policy thread the safe route.',
      cz: 'Jeden krok na stisk klávesy (tahová hra) — nespěchejte. Protože led klouže, volte cesty s '
        + 'odstupem od děr, ne těsně podél nich. Přepněte na „Sledovat AI“ a uvidíte, jak se naučená '
        + 'strategie protáhne bezpečnou cestou.',
    },
  },

  frozenlake_noslip: {
    goal: {
      en: 'The same frozen lake, but the ice does NOT slip — every move goes exactly where you point '
        + 'it. Find the single safe path from the start (top-left) to the goal flag (bottom-right), '
        + 'avoiding the holes. Reaching the goal scores 1; falling in a hole scores 0.',
      cz: 'Stejné jezero, ale led NEKLOUŽE — každý krok jde přesně tam, kam míříte. Najděte jedinou '
        + 'bezpečnou cestu ze startu (vlevo nahoře) k vlajce cíle (vpravo dole) a vyhněte se dírám. '
        + 'Dosažení cíle dá 1 bod; pád do díry dá 0.',
    },
    controls: [
      { keys: '← / A', action: { en: 'Move left', cz: 'Krok doleva' } },
      { keys: '→ / D', action: { en: 'Move right', cz: 'Krok doprava' } },
      { keys: '↑ / W', action: { en: 'Move up', cz: 'Krok nahoru' } },
      { keys: '↓ / S', action: { en: 'Move down', cz: 'Krok dolů' } },
    ],
    tips: {
      en: 'With no slipping this is a pure maze — plan a route around the holes and follow it exactly. '
        + 'It is the gentlest grid-world, so a trained agent solves it almost perfectly. Switch to '
        + '"Watch AI" to see the shortest safe path.',
      cz: 'Bez klouzání jde o čisté bludiště — naplánujte cestu kolem děr a držte se jí přesně. Je to '
        + 'nejjednodušší mřížkový svět, takže ho natrénovaný agent vyřeší téměř dokonale. Přepněte na '
        + '„Sledovat AI“ a uvidíte nejkratší bezpečnou cestu.',
    },
  },

  frozenlake8x8: {
    goal: {
      en: 'The bigger 8×8 frozen lake — more holes and a longer slippery path from the start '
        + '(top-left) to the goal flag (bottom-right). Same rules and the same slippery ice as the 4×4: '
        + 'you go where you press only about one time in three, the rest you slide sideways — over this '
        + 'longer path that adds up, so expect to be pushed off course often. Reaching the goal scores '
        + '1; falling in a hole or running out of moves scores 0.',
      cz: 'Větší zamrzlé jezero 8×8 — víc děr a delší kluzká cesta ze startu (vlevo nahoře) k vlajce '
        + 'cíle (vpravo dole). Stejná pravidla i stejně kluzký led jako u 4×4: tam, kam zmáčknete, '
        + 'půjdete jen asi jednou ze tří, jinak vás to smekne do strany — na delší cestě se to nasčítá, '
        + 'takže čekejte, že vás to bude často srážet z kurzu. Dosažení cíle dá 1 bod; pád do díry nebo '
        + 'vyčerpání tahů dá 0.',
    },
    controls: [
      { keys: '← / A', action: { en: 'Move left', cz: 'Krok doleva' } },
      { keys: '→ / D', action: { en: 'Move right', cz: 'Krok doprava' } },
      { keys: '↑ / W', action: { en: 'Move up', cz: 'Krok nahoru' } },
      { keys: '↓ / S', action: { en: 'Move down', cz: 'Krok dolů' } },
    ],
    tips: {
      en: 'A longer route means more chances for the ice to slip you off course, so keep clear of the '
        + 'holes and re-plan after each unexpected slide. Switch to "Watch AI" to see how a trained '
        + 'agent handles the bigger board.',
      cz: 'Delší cesta znamená víc příležitostí, aby vás led smekl z dráhy, takže se držte dál od děr a '
        + 'po každém nečekaném smyku cestu přeplánujte. Přepněte na „Sledovat AI“ a uvidíte, jak si s '
        + 'větší deskou poradí natrénovaný agent.',
    },
  },

  cliffwalking: {
    goal: {
      en: 'Walk from the start (bottom-left) to the goal flag (bottom-right) along the edge of the '
        + 'cliff (the red squares). Every step costs −1; stepping onto the cliff costs −100 and sends '
        + 'you back to the start. Reach the goal in as few steps as you can — the best route is about −13.',
      cz: 'Dojděte ze startu (vlevo dole) k vlajce cíle (vpravo dole) podél okraje útesu (červená '
        + 'políčka). Každý krok stojí −1; vstup na útes stojí −100 a vrátí vás na start. Dojděte do cíle '
        + 'na co nejméně kroků — nejlepší cesta je kolem −13.',
    },
    controls: [
      { keys: '↑ / W', action: { en: 'Move up', cz: 'Krok nahoru' } },
      { keys: '→ / D', action: { en: 'Move right', cz: 'Krok doprava' } },
      { keys: '↓ / S', action: { en: 'Move down', cz: 'Krok dolů' } },
      { keys: '← / A', action: { en: 'Move left', cz: 'Krok doleva' } },
    ],
    tips: {
      en: 'The shortest route runs one row above the cliff and straight across — risky but fast. A '
        + 'safer route climbs to the top row first, crosses, then drops to the goal. Switch to "Watch '
        + 'AI" to compare what a trained agent prefers.',
      cz: 'Nejkratší cesta vede jednu řadu nad útesem a rovnou napříč — riskantní, ale rychlá. '
        + 'Bezpečnější cesta nejdřív vystoupá do horní řady, přejde a teprve pak klesne do cíle. '
        + 'Přepněte na „Sledovat AI“ a porovnejte, co si vybere natrénovaný agent.',
    },
  },

  taxi: {
    goal: {
      en: 'Drive the taxi to the waiting passenger, pick them up, then drive to their destination flag '
        + 'and drop them off. Each step costs −1, a correct drop-off pays +20, and an illegal pickup or '
        + 'drop-off costs −10. A good run scores around +8 — so be quick and only pick up / drop off at '
        + 'the right stop.',
      cz: 'Dojeďte taxíkem k čekajícímu cestujícímu, naberte ho a pak dojeďte k vlajce jeho cíle a '
        + 'vysaďte ho. Každý krok stojí −1, správné vysazení vyplatí +20 a nelegální nabrání či vysazení '
        + 'stojí −10. Dobrý běh dá kolem +8 — buďte rychlí a nabírejte/vysazujte jen na správné zastávce.',
    },
    controls: [
      { keys: '↑ ↓ ← →', action: { en: 'Drive the taxi (also WASD)', cz: 'Řídit taxík (také WASD)' } },
      { keys: 'P / Space', action: { en: 'Pick up the passenger', cz: 'Nabrat cestujícího' } },
      { keys: 'O / Enter', action: { en: 'Drop off the passenger', cz: 'Vysadit cestujícího' } },
    ],
    tips: {
      en: 'Drive to the passenger\'s coloured stop first and press P to pick them up (they then ride in '
        + 'the taxi), then drive to the destination flag and press O to drop off. Picking up or dropping '
        + 'off anywhere else wastes 10 points. Switch to "Watch AI" to see an efficient route.',
      cz: 'Nejdřív dojeďte k barevné zastávce cestujícího a stiskem P ho naberte (pak jede v taxíku), '
        + 'poté dojeďte k vlajce cíle a stiskem O ho vysaďte. Nabrání či vysazení jinde stojí 10 bodů. '
        + 'Přepněte na „Sledovat AI“ a uvidíte úspornou trasu.',
    },
  },

  // ── MiniGrid (G2c) — turn-based, sparse-reward exploration grids ─────────────────────────────
  minigrid_empty: {
    goal: {
      en: 'Reach the green goal square in the small empty room. Turn to face the way you want to go, then '
        + 'step forward. The simplest level — no obstacles, just find the goal. You score only on reaching '
        + 'it, and the fewer steps you take the higher the score. (MiniGrid counts the surrounding wall in '
        + 'the grid size, so this "5×5" is a 3×3 room inside.)',
      cz: 'Dojděte na zelené cílové pole v malé prázdné místnosti. Otočte se směrem, kterým chcete jít, a '
        + 'pak udělejte krok vpřed. Nejjednodušší úroveň — žádné překážky, jen najděte cíl. Bodujete jen za '
        + 'jeho dosažení a čím méně kroků uděláte, tím vyšší skóre. (MiniGrid počítá do velikosti i '
        + 'obvodovou zeď, takže tahle „5×5“ je uvnitř místnost 3×3.)',
    },
    controls: MINIGRID_CONTROLS,
    tips: MINIGRID_TIPS,
  },

  minigrid_fourrooms: {
    goal: {
      en: 'Find and reach the green goal square in a layout of four rooms joined by narrow doorways. Your '
        + 'start and the goal are placed randomly each game, so you must explore from room to room through '
        + 'the gaps to find it. You score only on reaching the goal.',
      cz: 'Najděte a dosáhněte zeleného cílového pole v rozložení čtyř místností spojených úzkými průchody. '
        + 'Váš start i cíl jsou každou hru umístěny náhodně, takže musíte zkoumat z místnosti do místnosti '
        + 'přes průchody, abyste cíl našli. Bodujete jen za dosažení cíle.',
    },
    controls: MINIGRID_CONTROLS,
    tips: MINIGRID_TIPS,
  },

  minigrid_doorkey: {
    goal: {
      en: 'Pick up the key, unlock the door with it, then reach the green goal on the far side. Stand on '
        + 'the key and press P to grab it, face the locked door and press Space to open it, then walk '
        + 'through to the goal. You must do these in order — the door will not open without the key.',
      cz: 'Seberte klíč, odemkněte jím dveře a pak dojděte k zelenému cíli na druhé straně. Postavte se na '
        + 'klíč a stiskem P ho seberte, otočte se čelem k zamčeným dveřím a stiskem mezerníku je otevřete, '
        + 'pak projděte k cíli. Musíte to udělat v pořadí — bez klíče se dveře neotevřou.',
    },
    controls: MINIGRID_CONTROLS,
    tips: MINIGRID_TIPS,
  },

  minigrid_keycorridor: {
    goal: {
      en: 'Pick up the coloured ball that is locked inside a room. Explore the corridor to find the key '
        + '(it sits behind a door): face it and press P to grab it, then face the locked door and press '
        + 'Space to open it. Now the catch — you can carry only one thing at a time, so before you can '
        + 'pick up the ball you must face an empty square and press O to drop the key, then turn to face '
        + 'the ball and press P. The hardest of these four; it needs real, patient exploration.',
      cz: 'Seberte barevný míček zamčený uvnitř jedné místnosti. Prozkoumejte chodbu a najděte klíč '
        + '(je za dveřmi): otočte se k němu čelem a stiskem P ho seberte, pak se postavte čelem k '
        + 'zamčeným dveřím a stiskem mezerníku je otevřete. A teď ten háček — najednou unesete jen jednu '
        + 'věc, takže než seberete míček, musíte se otočit čelem na prázdné políčko a stiskem O klíč '
        + 'odložit, pak se otočit čelem k míčku a stisknout P. Nejtěžší ze čtyř; vyžaduje skutečné, '
        + 'trpělivé zkoumání.',
    },
    controls: MINIGRID_CONTROLS,
    tips: MINIGRID_TIPS,
  },
}

export const DEFAULT_PLAY_GUIDE: PlayGuide = PLAY_GUIDES.cartpole

// ── Atari (ALE) — one shared guide for the whole family (G4a) ──────────────────────────────────
// The 60+ Atari games share the exact same keyboard (full_action_space=True → fixed action indices,
// see content/playKeymaps.ts ATARI_KEYMAP) and the same arcade tips, so controls + tips are shared.
// The per-game *objective* is the env's own description (written once in the backend registry), so
// `atariPlayGuide(description)` slots it in as the goal — no per-game text duplicated on the client.
const ATARI_CONTROLS: PlayControl[] = [
  { keys: '↑ ↓ ← →', action: { en: 'Move (also WASD)', cz: 'Pohyb (také WASD)' } },
  { keys: 'Space', action: { en: 'Fire / action button', cz: 'Palba / akční tlačítko' } },
  { keys: '(release)', action: { en: 'Do nothing', cz: 'Nedělat nic' } },
]

const ATARI_TIPS: Bilingual = {
  en: 'These are real-time arcade games — if it feels too fast, lower the play speed (down to 0.1×) '
    + 'so you have time to react. Not every game uses Fire, and holding a direction together with '
    + 'Space combines them. The episode ends on game-over (lives lost, time up, or a win). Switch to '
    + '"Watch AI" once a model has been trained for this game on a GPU.',
  cz: 'Tohle jsou arkádové hry v reálném čase — pokud je to moc rychlé, snižte rychlost hraní (až na '
    + '0,1×), ať máte čas reagovat. Ne každá hra používá palbu a podržení směru spolu s mezerníkem je '
    + 'zkombinuje. Epizoda končí při „game over“ (ztráta životů, vypršení času nebo výhra). Až bude pro '
    + 'tuto hru natrénovaný model na GPU, přepněte na „Sledovat AI“.',
}

/** Build the play guide for an Atari env: its registry description is the goal; controls + tips are
 *  shared across the whole family. Keeps per-game prose in one place (the backend) instead of here. */
export function atariPlayGuide(description: Bilingual): PlayGuide {
  return { goal: description, controls: ATARI_CONTROLS, tips: ATARI_TIPS }
}

export function playGuideFor(envId: string | null): PlayGuide {
  return (envId !== null && PLAY_GUIDES[envId]) || DEFAULT_PLAY_GUIDE
}
