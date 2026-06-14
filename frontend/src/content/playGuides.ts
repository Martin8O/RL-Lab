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
}

export const DEFAULT_PLAY_GUIDE: PlayGuide = PLAY_GUIDES.cartpole

export function playGuideFor(envId: string | null): PlayGuide {
  return (envId !== null && PLAY_GUIDES[envId]) || DEFAULT_PLAY_GUIDE
}
