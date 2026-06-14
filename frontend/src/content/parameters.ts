// Parameter info content (B5) — beginner-friendly, bilingual (CZ/EN) explanations
// for every tunable in the sidebar. This is the data-driven "learning aid" feature:
// each sidebar control opens a popup that reads its entry from here.
//
// Readability convention (2026-06-13): the `general` (and multi-concept) texts use a tiny markup
// rendered by ParamInfo — **term** is bold, and "\n" starts a new line — so each popup leads with a
// bold key term and multi-concept popups read as a scannable list, not one wall of text.
//
// Extensibility: adding a new parameter = adding one entry below (no UI/logic change).
// Adding a new game = adding a `perEnv[<envId>]` note to each relevant parameter.
//
// Cross-checked against the source cookbook's recommended PPO defaults:
//   lr 3e-4 · γ 0.99 · clip 0.2 · ent 0.0 · 2×64 net · tanh.

import type { Bilingual } from '../api/types'

export interface ParamInfo {
  /** What it is, what moving it does, and when to touch it — the core explanation. */
  general: Bilingual
  /** Beginner-friendly guidance on the recommended value. Omit for read-only metrics. */
  recommended?: Bilingual
  /** Typical usable range, shown as a language-neutral string (e.g. "1e-5 – 1e-2"). Omit for metrics. */
  range?: string
  /** Per-environment note, keyed by env id (e.g. "cartpole"). Omit for env-agnostic concepts. */
  perEnv?: Record<string, Bilingual>
}

export const PARAM_INFO: Record<string, ParamInfo> = {
  algorithm: {
    general: {
      en: '**How the agent learns.**\n**PPO** (reinforcement learning) — tweaks one neural network with gradients after each batch of play; steady and sample-efficient.\n**Neuroevolution** — keeps a whole population of networks, scores them, and breeds the best (mutation + crossover) each generation; simple, gradient-free, like "survival of the fittest".',
      cz: '**Jak se agent učí.**\n**PPO** (zpětnovazební učení) — upravuje jednu neuronovou síť pomocí gradientů po každé dávce hraní; stabilní a úsporné na data.\n**Neuroevoluce** — udržuje celou populaci sítí, ohodnotí je a v každé generaci množí ty nejlepší (mutace + křížení); jednoduchá, bez gradientů, jako „přežití nejschopnějších“.',
    },
    recommended: {
      en: 'PPO — the reliable, general-purpose default that also scales to harder games. Neuroevolution is gradient-free and can be surprisingly fast on simple tasks — try both and compare (see the per-game note below).',
      cz: 'PPO — spolehlivá, univerzální volba, která zvládne i těžší hry. Neuroevoluce je bezgradientní a u jednoduchých úloh bývá překvapivě rychlá — vyzkoušejte oba a porovnejte (viz poznámka k dané hře níže).',
    },
    perEnv: {
      cartpole: {
        en: 'Both solve CartPole. Counter-intuitively, Neuroevolution is often the faster of the two here — within a few wall-clock seconds — because the network is tiny and there are no gradient updates, just scoring a population. PPO is steadier and is the method that scales to hard problems, but on this easy task it usually needs more wall-clock time (it collects rollouts and runs gradient steps).',
        cz: 'Oba CartPole vyřeší. Možná překvapivě bývá tady rychlejší Neuroevoluce — během pár vteřin reálného času — protože síť je drobná a nejsou žádné gradientní úpravy, jen ohodnocení populace. PPO je stabilnější a je to metoda, která zvládne i těžké úlohy, ale u téhle snadné úlohy obvykle potřebuje víc reálného času (sbírá data a počítá gradientní kroky).',
      },
      lunarlander: {
        en: 'Both can be selected, but PPO is the practical choice here. LunarLander is much harder than CartPole, and PPO with enough steps (a few hundred thousand) reliably learns to land. The simple neuroevolution here uses a tiny network, so it improves but rarely reaches a clean landing within a few dozen generations.',
        cz: 'Vybrat lze obě, ale PPO je tu praktická volba. LunarLander je mnohem těžší než CartPole a PPO se s dostatkem kroků (řádově statisíce) spolehlivě naučí přistávat. Zdejší jednoduchá neuroevoluce používá drobnou síť, takže se sice zlepšuje, ale za pár desítek generací málokdy dosáhne čistého přistání.',
      },
      mountaincar: {
        en: 'A famous exploration trap. PPO with the default settings often stalls near −200: a random agent almost never reaches the flag, so there is no reward signal to learn from (raising the Entropy bonus helps it explore). Neuroevolution can do better here — among a whole population, some network stumbles onto the rocking motion that reaches the flag, and that success spreads. A great task for comparing the two.',
        cz: 'Pověstná past na zkoumání. PPO s výchozím nastavením často uvázne kolem −200: náhodný agent skoro nikdy nedojede k vlajce, takže není z čeho se učit (pomáhá zvýšit bonus za entropii). Neuroevoluce tu může být lepší — v celé populaci některá síť náhodou objeví houpavý pohyb k vlajce a úspěch se rozšíří. Skvělá úloha pro porovnání obou.',
      },
      acrobot: {
        en: 'Both handle Acrobot well. PPO reliably learns the pumping motion within a few hundred thousand steps and reaches the target (around −100). Neuroevolution also climbs steadily, since even a tiny network can discover the rhythmic swing. A good task to watch both succeed.',
        cz: 'Acrobot zvládnou obě. PPO se za pár set tisíc kroků spolehlivě naučí „pumpovat“ a dosáhne cíle (kolem −100). Neuroevoluce se tu také plynule zlepšuje, protože i drobná síť objeví rytmické houpání. Pěkná úloha, kde uspějí obě.',
      },
    },
  },

  learning_rate: {
    general: {
      en: '**Step size.** How big a step the agent takes when it updates its strategy after each batch of experience.\nToo high → learning becomes unstable and "forgets" what worked. Too low → it crawls, needing far more steps to improve.',
      cz: '**Velikost kroku.** Jak velký krok agent udělá při úpravě své strategie po každé dávce zkušeností.\nPříliš vysoká → nestabilita a „zapomínání“ toho, co fungovalo. Příliš nízká → učení se vleče a vyžaduje mnohem více kroků.',
    },
    recommended: {
      en: '3e-4 — a safe, steady default for PPO.',
      cz: '3e-4 — bezpečná, stabilní výchozí hodnota pro PPO.',
    },
    range: '1e-5 – 1e-2',
    perEnv: {
      cartpole: {
        en: 'CartPole is easy, so 3e-4 solves it in seconds. Pushing it higher can look faster but often collapses before reaching 500.',
        cz: 'CartPole je snadný, takže 3e-4 ho vyřeší během vteřin. Vyšší hodnota může vypadat rychleji, ale často zkolabuje, než dosáhne 500.',
      },
      lunarlander: {
        en: 'The default 3e-4 is a solid starting point for LunarLander too. It is a harder task, so progress is slower — give it more steps rather than a much higher learning rate, which tends to destabilise landing.',
        cz: 'Výchozí 3e-4 je dobrý výchozí bod i pro LunarLander. Je to těžší úloha, takže pokrok je pomalejší — dejte mu raději více kroků než výrazně vyšší rychlost učení, která přistávání spíše rozhází.',
      },
      mountaincar: {
        en: 'The default 3e-4 is fine. The real bottleneck on MountainCar is exploration, not step size — if PPO is stuck at −200, raise the Entropy bonus before touching the learning rate.',
        cz: 'Výchozí 3e-4 je v pořádku. Skutečnou překážkou u MountainCar je zkoumání, ne velikost kroku — pokud PPO uvázne na −200, zvyšte raději bonus za entropii než rychlost učení.',
      },
      acrobot: {
        en: '3e-4 works well for Acrobot. It is a stable task to learn, so there is little reason to change it — give it more steps rather than a higher rate.',
        cz: '3e-4 funguje pro Acrobot dobře. Je to stabilní úloha, takže není důvod ji měnit — dejte mu raději víc kroků než vyšší rychlost.',
      },
    },
  },

  gamma: {
    general: {
      en: '**Discount factor** — how much the agent values future rewards versus immediate ones.\nNear 1.0 → it plans far ahead. Lower → short-sighted, caring mostly about the next few steps.',
      cz: '**Diskontní faktor** — jak moc agent cení budoucí odměny oproti okamžitým.\nBlízko 1.0 → plánuje daleko dopředu. Nižší → krátkozraký, soustředí se hlavně na nejbližší kroky.',
    },
    recommended: {
      en: '0.99 — the standard for most control tasks.',
      cz: '0.99 — standard pro většinu řídicích úloh.',
    },
    range: '0.90 – 0.999',
    perEnv: {
      cartpole: {
        en: 'Balancing the pole is all about the long run, so a high γ (0.99) works best. Too low and the agent won\'t "see" that a small lean now means a fall later.',
        cz: 'Udržení tyče je hlavně o dlouhodobém výhledu, takže vysoké γ (0.99) funguje nejlépe. Při nízkém γ agent „neuvidí“, že malý náklon teď znamená pád později.',
      },
      lunarlander: {
        en: 'A landing is a long sequence of small thrusts, so the agent must value future reward highly — keep γ at 0.99. Too low and it fires the engines greedily instead of planning a gentle touchdown.',
        cz: 'Přistání je dlouhá řada malých zážehů, takže agent musí vysoko cenit budoucí odměnu — nechte γ na 0.99. Při nízkém γ pálí trysky hltavě místo plánování jemného dosednutí.',
      },
      mountaincar: {
        en: 'Reaching the flag is a long build-up, so keep γ high (0.99): the agent must value a distant success over the −1 it pays every step. Too low and it never "sees" that rocking back now pays off later.',
        cz: 'Dojezd k vlajce je dlouhé rozhoupávání, takže nechte γ vysoké (0.99): agent musí cenit vzdálený úspěch víc než −1, které platí každý krok. Při nízkém γ nikdy „neuvidí“, že couvnutí teď se vyplatí později.',
      },
      acrobot: {
        en: 'A swing-up is a long sequence too, so 0.99 is right. The agent must plan several swings ahead rather than chase the next step.',
        cz: 'Vyhoupnutí je také dlouhá řada kroků, takže 0.99 je správně. Agent musí plánovat několik kmitů dopředu, ne honit jen další krok.',
      },
    },
  },

  clip_range: {
    general: {
      en: "**PPO's safety rail.** It limits how far the new strategy can move from the old one in a single update, preventing one bad batch from wrecking a good policy.",
      cz: '**Bezpečnostní zábradlí PPO.** Omezuje, jak daleko se nová strategie může v jedné aktualizaci vzdálit od staré, aby jedna špatná dávka nezničila dobrou strategii.',
    },
    recommended: {
      en: '0.2 — the value from the PPO paper, rarely worth changing.',
      cz: '0.2 — hodnota z původního článku o PPO, jen zřídka stojí za změnu.',
    },
    range: '0.1 – 0.4',
    perEnv: {
      cartpole: {
        en: "0.2 is plenty for CartPole; there's little reason to touch it here.",
        cz: 'Pro CartPole je 0.2 bohatě dostačující; není důvod ji zde měnit.',
      },
      lunarlander: {
        en: '0.2 works well for LunarLander as well; there is rarely a reason to change it. Leave it and tune steps and learning rate first.',
        cz: '0.2 funguje dobře i pro LunarLander; jen zřídka je důvod ji měnit. Nechte ji a laďte nejdřív počet kroků a rychlost učení.',
      },
      mountaincar: {
        en: '0.2 is fine — clipping is not what holds MountainCar back. Leave it and adjust exploration (Entropy) and steps instead.',
        cz: '0.2 je v pořádku — ořezávání není to, co MountainCar brzdí. Nechte ji a laďte raději zkoumání (entropii) a počet kroků.',
      },
      acrobot: {
        en: '0.2 works well; there is rarely a reason to change it on Acrobot.',
        cz: '0.2 funguje dobře; u Acrobotu jen zřídka je důvod ji měnit.',
      },
    },
  },

  ent_coef: {
    general: {
      en: '**Entropy bonus** — rewards the agent for staying a bit random and exploring.\nRaise it if the agent settles too early on a mediocre habit. Keep it near 0 when the task is simple enough to solve by exploiting what it learns.',
      cz: '**Bonus za entropii** — odměňuje agenta za to, že zůstane trochu náhodný a zkoumá.\nZvyšte ho, pokud agent příliš brzy zapadne do průměrného návyku. Nechte ho u 0, když je úloha dost jednoduchá na vyřešení tím, co se naučil.',
    },
    recommended: {
      en: '0.0 for simple tasks; nudge it up (e.g. 0.01) if the agent settles too early on a mediocre habit.',
      cz: '0.0 pro jednoduché úlohy; mírně zvyšte (např. 0.01), pokud agent příliš brzy zapadne do průměrného návyku.',
    },
    range: '0.0 – 0.05',
    perEnv: {
      cartpole: {
        en: "CartPole's two actions are easy to explore, so 0 is fine. A tiny value (0.01) rarely hurts if learning stalls.",
        cz: 'Dvě akce CartPole se zkoumají snadno, takže 0 stačí. Drobná hodnota (0.01) málokdy uškodí, pokud učení uvázne.',
      },
      lunarlander: {
        en: 'LunarLander benefits from a little exploration early on, so a small value (e.g. 0.01) can help the agent discover the side thrusters before settling. 0.0 still works but may learn more slowly.',
        cz: 'LunarLanderu zpočátku pomáhá trocha zkoumání, takže drobná hodnota (např. 0.01) může agentovi pomoci objevit boční trysky dřív, než se ustálí. 0.0 také funguje, ale učení může být pomalejší.',
      },
      mountaincar: {
        en: 'This is the key dial for MountainCar. The reward is sparse, so a little extra exploration is often what lets PPO discover the flag at all — try raising it (e.g. 0.01–0.05) if the score is stuck at −200.',
        cz: 'U MountainCar je tohle klíčový knoflík. Odměna je řídká, takže trocha zkoumání navíc je často to, co PPO vůbec umožní objevit vlajku — zkuste ji zvýšit (např. 0.01–0.05), pokud skóre vězí na −200.',
      },
      acrobot: {
        en: 'A small value can speed up the early search for the pumping motion, but Acrobot is learnable at 0 as well. Nudge it to ~0.01 if progress stalls.',
        cz: 'Malá hodnota může urychlit počáteční hledání „pumpování“, ale Acrobot je řešitelný i při 0. Posuňte ji k ~0.01, pokud pokrok vázne.',
      },
    },
  },

  n_hidden_layers: {
    general: {
      en: "**Network depth** — how many stacked layers the agent's neural network has.\nMore layers can capture more complex patterns but train slower and may overfit a simple task.",
      cz: '**Hloubka sítě** — kolik vrstev má neuronová síť agenta.\nVíce vrstev zachytí složitější vzory, ale trénují se pomaleji a na jednoduché úloze se mohou přeučit.',
    },
    recommended: {
      en: '2 — the standard small network for control tasks.',
      cz: '2 — standardní malá síť pro řídicí úlohy.',
    },
    range: '1 – 4',
    perEnv: {
      cartpole: {
        en: "CartPole's 4-number observation is simple; 2 layers is more than enough. Extra depth just slows things down.",
        cz: 'Pozorování CartPole má jen 4 čísla; 2 vrstvy bohatě stačí. Větší hloubka jen vše zpomalí.',
      },
      lunarlander: {
        en: 'The 8-number observation is still small, so 2 layers is a fine default. LunarLander is harder than CartPole, but depth helps less than simply training for more steps.',
        cz: 'Pozorování o osmi číslech je stále malé, takže 2 vrstvy jsou dobrá výchozí volba. LunarLander je těžší než CartPole, ale hloubka pomáhá méně než prostě delší trénink.',
      },
      mountaincar: {
        en: 'The 2-number observation is tiny, so 2 layers is ample. Extra depth will not fix MountainCar — exploration (Entropy) will.',
        cz: 'Pozorování o dvou číslech je drobné, takže 2 vrstvy bohatě stačí. Hloubka navíc MountainCar nevyřeší — vyřeší ho zkoumání (entropie).',
      },
      acrobot: {
        en: 'The 6-number observation is still small; 2 layers is plenty for Acrobot.',
        cz: 'Pozorování o šesti číslech je stále malé; pro Acrobot 2 vrstvy bohatě stačí.',
      },
    },
  },

  neurons_per_layer: {
    general: {
      en: '**Network width** — how many neurons each layer has.\nWider networks have more capacity but cost more compute; for tiny observations a narrow net learns just as well.',
      cz: '**Šířka sítě** — kolik neuronů má každá vrstva.\nŠirší sítě mají větší kapacitu, ale stojí více výpočtu; pro malá pozorování se úzká síť naučí stejně dobře.',
    },
    recommended: {
      en: '64 — paired with 2 layers, the classic 2×64 network.',
      cz: '64 — spolu s 2 vrstvami klasická síť 2×64.',
    },
    range: '16 – 256',
    perEnv: {
      cartpole: {
        en: '64 neurons easily handle CartPole. Going wider gives no benefit and just uses more CPU.',
        cz: '64 neuronů CartPole hravě zvládne. Více již nepřinese žádnou výhodu, jen spotřebuje více CPU.',
      },
      lunarlander: {
        en: '64 neurons handle LunarLander well. You can try 128 for a little more capacity on this harder task, but more steps usually helps more than a wider net.',
        cz: '64 neuronů zvládne LunarLander dobře. U této těžší úlohy můžete zkusit 128 pro trochu větší kapacitu, ale víc kroků obvykle pomůže víc než širší síť.',
      },
      mountaincar: {
        en: "64 neurons easily cover MountainCar's tiny state. Going wider just costs CPU without helping.",
        cz: '64 neuronů drobný stav MountainCar hravě pokryje. Širší síť jen spotřebuje CPU, aniž by pomohla.',
      },
      acrobot: {
        en: '64 neurons handle Acrobot well; there is no need to widen the net here.',
        cz: '64 neuronů Acrobot zvládne dobře; síť tu není potřeba rozšiřovat.',
      },
    },
  },

  activation: {
    general: {
      en: "**Activation function** — the non-linearity each neuron applies, letting the network learn curved, non-trivial behaviour.\n**tanh** — smooth and stable (PPO's default). **relu** — faster, common in larger or vision networks.",
      cz: '**Aktivační funkce** — nelinearita, kterou každý neuron používá, aby se síť naučila zakřivené, netriviální chování.\n**tanh** — hladká a stabilní (výchozí pro PPO). **relu** — rychlejší, běžná u větších či obrazových sítí.',
    },
    recommended: {
      en: "tanh — SB3's PPO default, stable on control tasks.",
      cz: 'tanh — výchozí pro PPO v SB3, stabilní u řídicích úloh.',
    },
    range: 'tanh / relu',
    perEnv: {
      cartpole: {
        en: 'tanh is the proven choice for CartPole. relu works too but offers no real advantage on such a small net.',
        cz: 'tanh je pro CartPole osvědčená volba. relu také funguje, ale na tak malé síti nepřináší skutečnou výhodu.',
      },
      lunarlander: {
        en: 'tanh is a safe default here too. relu also works on LunarLander and is worth a try with a wider network, but the difference is small.',
        cz: 'tanh je i tady bezpečná výchozí volba. relu na LunarLanderu také funguje a u širší sítě stojí za zkoušku, ale rozdíl je malý.',
      },
      mountaincar: {
        en: 'tanh is a safe default. The activation is not the lever on MountainCar — exploration is.',
        cz: 'tanh je bezpečná výchozí volba. Aktivační funkce není u MountainCar ta páka — tou je zkoumání.',
      },
      acrobot: {
        en: 'tanh is the proven choice; relu works too but makes little difference on such a small net.',
        cz: 'tanh je osvědčená volba; relu také funguje, ale na tak malé síti je rozdíl malý.',
      },
    },
  },

  seed: {
    general: {
      en: '**Random seed** — the starting point for all randomness: environment resets, network initialisation, action sampling.\nFixing it makes a run reproducible: same seed + same settings = same result, essential for fair comparisons.',
      cz: '**Náhodné semeno** — výchozí bod pro veškerou náhodnost: resety prostředí, inicializaci sítě, výběr akcí.\nPevné semeno zajistí reprodukovatelnost: stejné semeno + stejná nastavení = stejný výsledek, klíčové pro férové porovnání.',
    },
    recommended: {
      en: 'Any fixed integer (e.g. 42). Change it to see how much luck affects a run.',
      cz: 'Libovolné pevné číslo (např. 42). Změňte ho, abyste viděli, jak moc běh ovlivní náhoda.',
    },
    range: '0 – 999999',
    perEnv: {
      cartpole: {
        en: 'CartPole solves on most seeds, but some converge faster than others — handy for seeing run-to-run variance.',
        cz: 'CartPole se vyřeší při většině semen, ale některá konvergují rychleji než jiná — užitečné pro sledování rozptylu mezi běhy.',
      },
      lunarlander: {
        en: 'LunarLander varies more between seeds than CartPole — the terrain and start differ — so the same settings can land on one seed and crash on another. Fix it for fair comparisons; change it to gauge how robust a setup really is.',
        cz: 'LunarLander se mezi semeny liší víc než CartPole — terén i start jsou jiné — takže stejná nastavení mohou na jednom semenu přistát a na jiném havarovat. Pro férové porovnání ho zafixujte; změnou zjistíte, jak robustní nastavení doopravdy je.',
      },
      mountaincar: {
        en: 'Luck matters a lot here — on some seeds a run stumbles onto the flag early and takes off, on others it never does. Fix it for fair comparisons; change it to feel how much exploration depends on chance.',
        cz: 'Náhoda tu hraje velkou roli — na některých semenech běh brzy narazí na vlajku a rozjede se, na jiných nikdy. Pro férové porovnání ho zafixujte; změnou ucítíte, jak moc zkoumání závisí na štěstí.',
      },
      acrobot: {
        en: 'Acrobot is fairly robust across seeds, but fixing it still keeps runs reproducible and comparisons fair.',
        cz: 'Acrobot je vůči semenům poměrně robustní, ale zafixování přesto udrží běhy reprodukovatelné a porovnání férové.',
      },
    },
  },

  total_steps: {
    general: {
      en: '**Training budget** — how many environment steps the agent gets to learn from before stopping.\nMore steps mean more practice and usually a better policy, up to the point where it has mastered the task.',
      cz: '**Tréninkový rozpočet** — kolik kroků v prostředí agent dostane na učení, než se zastaví.\nVíce kroků znamená více cviku a obvykle lepší strategii, dokud úlohu nezvládne.',
    },
    recommended: {
      en: 'Enough steps for the agent to master the task — easy games need only a little, harder ones far more (see the per-game note).',
      cz: 'Tolik kroků, aby agent úlohu zvládl — snadné hry potřebují málo, těžší mnohem víc (viz poznámka k dané hře).',
    },
    range: '10k – 2M (depends on the game)',
    perEnv: {
      cartpole: {
        en: 'CartPole is usually solved by ~30–50k steps. Beyond that the reward just sits at 500 — more steps won\'t help.',
        cz: 'CartPole se obvykle vyřeší kolem 30–50k kroků. Dál už odměna jen sedí na 500 — více kroků nepomůže.',
      },
      lunarlander: {
        en: 'LunarLander needs far more practice than CartPole — pick the largest option (200k) for a visible landing policy, and on a faster machine even more (~0.5–1M) is ideal. The 50k ★ is tuned for CartPole; for LunarLander treat it as a bare minimum.',
        cz: 'LunarLander potřebuje mnohem víc cviku než CartPole — zvolte největší možnost (200k) pro viditelnou strategii přistávání a na výkonnějším stroji je ideální i víc (~0,5–1M). Hvězdička u 50k je laděná pro CartPole; pro LunarLander ji berte jako naprosté minimum.',
      },
      mountaincar: {
        en: 'More steps alone may not solve MountainCar — with too little exploration the score sits at −200 no matter the budget. Pair a healthy step count with a higher Entropy bonus, or try Neuroevolution.',
        cz: 'Samotné kroky navíc MountainCar nemusí vyřešit — při příliš malém zkoumání skóre vězí na −200 bez ohledu na rozpočet. Spojte rozumný počet kroků s vyšším bonusem za entropii, nebo zkuste Neuroevoluci.',
      },
      acrobot: {
        en: 'Acrobot usually solves within a few hundred thousand steps, so the 200k ★ is a good budget. Give it more if it has not reached the target yet.',
        cz: 'Acrobot se obvykle vyřeší za pár set tisíc kroků, takže 200k (★) je dobrý rozpočet. Dejte mu víc, pokud ještě nedosáhl cíle.',
      },
    },
  },

  // ── Neuroevolution settings (C2) ──────────────────────────────────────────
  // The "200 cars" idea: instead of one agent improving by gradient steps, a whole
  // population of networks is scored each generation and the best are bred together.

  population_size: {
    general: {
      en: '**Herd size** — how many different networks are tried in each generation.\nA bigger population explores more strategies at once and is more likely to find a good one, but every generation takes proportionally longer to score.',
      cz: '**Velikost stáda** — kolik různých sítí se vyzkouší v každé generaci.\nVětší populace zkouší více strategií najednou a má větší šanci najít dobrou, ale ohodnocení každé generace trvá úměrně déle.',
    },
    recommended: {
      en: '50 — plenty of variety while still scoring a generation in a second or two on CPU.',
      cz: '50 — dostatek rozmanitosti, a přitom se generace na CPU ohodnotí za vteřinu či dvě.',
    },
    range: '10 – 200',
    perEnv: {
      cartpole: {
        en: 'CartPole is easy, so even 30–50 networks find a balancing strategy within a few generations. Larger populations mostly just cost more CPU here.',
        cz: 'CartPole je snadný, takže i 30–50 sítí najde strategii pro udržení tyče během několika generací. Větší populace zde většinou jen spotřebují více CPU.',
      },
      lunarlander: {
        en: 'A bigger population explores more landing strategies per generation, so on this harder task leaning toward the upper end (e.g. 100) can help — at a proportional cost in time per generation.',
        cz: 'Větší populace zkouší za generaci víc strategií přistávání, takže u této těžší úlohy může pomoci posun k horní hranici (např. 100) — za úměrnou cenu v čase na generaci.',
      },
      mountaincar: {
        en: 'A bigger population is a real advantage here: with more networks tried per generation, one is far more likely to stumble onto the flag and seed the whole herd. Lean toward the upper end (e.g. 100).',
        cz: 'Větší populace je tu skutečná výhoda: s víc sítěmi za generaci je mnohem pravděpodobnější, že některá narazí na vlajku a „nasází“ celé stádo. Posuňte se k horní hranici (např. 100).',
      },
      acrobot: {
        en: '50 networks find the swing-up within a reasonable number of generations; a larger population helps a little, at a proportional cost in time.',
        cz: '50 sítí najde vyhoupnutí za rozumný počet generací; větší populace pomůže o něco víc, za úměrnou cenu v čase.',
      },
    },
  },

  top_k_parents: {
    general: {
      en: '**Survivors** — how many of the best performers become parents of the next generation.\nToo few → the gene pool narrows fast (everyone ends up alike). Too many → weak networks keep breeding, slowing progress.',
      cz: '**Přeživší** — kolik nejlepších jedinců se stane rodiči další generace.\nPříliš málo → genofond se rychle zúží (všichni si jsou podobní). Příliš mnoho → množí se i slabé sítě a pokrok se zpomalí.',
    },
    recommended: {
      en: '10 — the top fifth of a 50-network population; a healthy balance of quality and diversity.',
      cz: '10 — horní pětina z populace 50 sítí; zdravá rovnováha kvality a rozmanitosti.',
    },
    range: '2 – 50',
    perEnv: {
      cartpole: {
        en: 'Keeping the top ~10 works well for CartPole. Setting it to 2 can lock in one early lucky strategy before it has truly mastered balancing.',
        cz: 'Ponechání horních ~10 funguje pro CartPole dobře. Nastavení na 2 může zafixovat jednu časnou šťastnou strategii dříve, než skutečně zvládne balancování.',
      },
      lunarlander: {
        en: 'Keeping the top ~10 keeps enough diversity for LunarLander. Too few parents and the herd locks onto one mediocre descent before it learns to land softly.',
        cz: 'Ponechání horních ~10 udrží pro LunarLander dost rozmanitosti. Při příliš málo rodičích se stádo upne na jeden průměrný sestup dřív, než se naučí měkce přistát.',
      },
      mountaincar: {
        en: 'Keep enough parents (~10) so one lucky flag-reacher does not dominate before the strategy is reliable. Too few and the herd narrows onto a single fluke.',
        cz: 'Ponechte dost rodičů (~10), aby jeden šťastlivec u vlajky neovládl populaci dřív, než je strategie spolehlivá. Příliš málo a stádo se zúží na jednu náhodu.',
      },
      acrobot: {
        en: 'The top ~10 keeps enough diversity for Acrobot to refine the pumping motion across generations.',
        cz: 'Horních ~10 udrží pro Acrobot dost rozmanitosti, aby napříč generacemi vyladil „pumpování“.',
      },
    },
  },

  mutation_rate: {
    general: {
      en: "**Creative noise** — how strongly each offspring's weights are randomly nudged when it is born.\nHigher → explores boldly but unpredictably. Lower → fine-tunes carefully but can get stuck on a so-so strategy.",
      cz: '**Tvůrčí šum** — jak silně se náhodně pozmění váhy každého potomka při „narození“.\nVyšší → zkoumá odvážně, ale nepředvídatelně. Nižší → pečlivě dolaďuje, ale může uvíznout u průměrné strategie.',
    },
    recommended: {
      en: '0.1 — small, steady tweaks that improve the herd without scrambling good networks.',
      cz: '0.1 — malé, stálé úpravy, které zlepšují stádo, aniž by rozházely dobré sítě.',
    },
    range: '0.01 – 1.0',
    perEnv: {
      cartpole: {
        en: 'For CartPole 0.1 climbs to 500 reliably. Push it toward 1.0 and good balancers get scrambled each generation, so the best score jumps around instead of settling.',
        cz: 'U CartPole 0.1 spolehlivě vyšplhá k 500. Posuňte ji k 1.0 a dobré „balancéry“ se každou generaci rozházejí, takže nejlepší skóre poskakuje místo ustálení.',
      },
      lunarlander: {
        en: 'On this harder task a moderate 0.1 still balances exploration and stability. Too high and good descents get scrambled every generation; too low and the herd struggles to discover a working landing at all.',
        cz: 'U této těžší úlohy stále vyvažuje umírněných 0.1 zkoumání a stabilitu. Příliš vysoká a dobré sestupy se každou generaci rozházejí; příliš nízká a stádo vůbec těžko objeví funkční přistání.',
      },
      mountaincar: {
        en: 'A slightly higher rate can help the population explore enough to find the flag at all; once some networks reach it, 0.1 refines them well.',
        cz: 'Mírně vyšší míra může populaci pomoci dostatečně zkoumat, aby vlajku vůbec našla; jakmile k ní některé sítě dojedou, 0.1 je dobře doladí.',
      },
      acrobot: {
        en: '0.1 balances exploration and stability for Acrobot. Too high and good swingers get scrambled every generation.',
        cz: '0.1 vyvažuje u Acrobotu zkoumání a stabilitu. Příliš vysoká a dobří „houpači“ se každou generaci rozházejí.',
      },
    },
  },

  crossover_rate: {
    general: {
      en: '**Parent mixing** — the chance a new network blends two parents (combining their strengths) rather than copying one.\nCrossover spreads ideas across the population; at 0 each child descends from a single parent plus mutation.',
      cz: '**Mísení rodičů** — pravděpodobnost, že nová síť mísí dva rodiče (kombinuje jejich přednosti) místo kopie jednoho.\nKřížení šíří nápady napříč populací; při 0 každý potomek pochází jen z jednoho rodiče plus mutace.',
    },
    recommended: {
      en: '0.5 — half the offspring blend two parents, half refine one; a common, well-rounded default.',
      cz: '0.5 — polovina potomků mísí dva rodiče, polovina dolaďuje jednoho; běžná, vyvážená výchozí hodnota.',
    },
    range: '0.0 – 1.0',
    perEnv: {
      cartpole: {
        en: 'CartPole solves across a wide range of crossover rates, so this is a safe dial to experiment with and watch how mixing parents affects the climb.',
        cz: 'CartPole se vyřeší v širokém rozsahu měr křížení, takže je to bezpečný knoflík k experimentování a sledování, jak mísení rodičů ovlivňuje vzestup.',
      },
      lunarlander: {
        en: 'Mixing parents spreads useful descent tricks through the population; 0.5 is a safe default for LunarLander. Feel free to experiment — it rarely breaks a run.',
        cz: 'Mísení rodičů šíří užitečné triky sestupu napříč populací; 0.5 je pro LunarLander bezpečná výchozí hodnota. Klidně experimentujte — běh to málokdy pokazí.',
      },
      mountaincar: {
        en: 'Mixing parents spreads a rare flag-reaching trick through the population, so 0.5 is a safe default — most useful once at least one network has succeeded.',
        cz: 'Mísení rodičů šíří vzácný trik, jak dojet k vlajce, napříč populací, takže 0.5 je bezpečná výchozí hodnota — nejužitečnější, jakmile uspěje aspoň jedna síť.',
      },
      acrobot: {
        en: '0.5 is a safe default for Acrobot; mixing parents spreads the pumping rhythm through the herd.',
        cz: '0.5 je pro Acrobot bezpečná výchozí hodnota; mísení rodičů šíří rytmus „pumpování“ napříč stádem.',
      },
    },
  },

  generations: {
    general: {
      en: '**Breeding rounds** — the evolution equivalent of training length.\nEach generation scores the whole population, keeps the best, and breeds the next. More generations give more chances to refine, up to the point the task is mastered.',
      cz: '**Kola šlechtění** — evoluční obdoba délky tréninku.\nKaždá generace ohodnotí celou populaci, ponechá nejlepší a vyšlechtí další. Více generací dává více příležitostí k vylepšení, dokud úloha není zvládnuta.',
    },
    recommended: {
      en: '30 is a solid default for simple tasks; harder games need many more generations (see the per-game note).',
      cz: '30 je dobrá výchozí hodnota pro jednoduché úlohy; těžší hry potřebují mnohem víc generací (viz poznámka k dané hře).',
    },
    range: '5 – 200',
    perEnv: {
      cartpole: {
        en: 'CartPole is often essentially solved within 10–20 generations; beyond that the best fitness just sits near 500, much like extra steps do for PPO.',
        cz: 'CartPole bývá v podstatě vyřešen během 10–20 generací; dále už nejlepší fitness jen sedí poblíž 500, podobně jako kroky navíc u PPO.',
      },
      lunarlander: {
        en: 'LunarLander is hard for the tiny evolution network, so it needs many more generations than CartPole — and may still not fully solve it. Push this high (100+) to watch it keep improving; PPO is the faster route to an actual landing.',
        cz: 'LunarLander je pro drobnou evoluční síť těžký, takže potřebuje mnohem víc generací než CartPole — a stejně ho nemusí plně vyřešit. Posuňte hodnotu vysoko (100+), pokud chcete sledovat další zlepšování; PPO je rychlejší cesta ke skutečnému přistání.',
      },
      mountaincar: {
        en: 'It can take many generations before a network first reaches the flag — then progress accelerates. Give it plenty (e.g. 50+) and a larger population to improve the odds.',
        cz: 'Než nějaká síť poprvé dojede k vlajce, může to trvat řadu generací — pak se pokrok zrychlí. Dejte jí dost (např. 50+) a větší populaci, ať zvýšíte šance.',
      },
      acrobot: {
        en: 'Acrobot usually improves steadily over a few dozen generations; push higher to keep refining the swing-up.',
        cz: 'Acrobot se obvykle plynule zlepšuje během pár desítek generací; vyšší hodnotou ho necháte dál ladit vyhoupnutí.',
      },
    },
  },

  // ── Chart concepts (B5 follow-up) ─────────────────────────────────────────
  // The reward/loss/fitness tabs and the Smooth control. These describe what a
  // curve means rather than a tunable, so most omit recommended/range.

  reward: {
    general: {
      en: "**Average score per episode** — the headline number for how well the agent is playing.\nThe agent's entire objective is to push this curve upward over training, so a rising reward means it is learning.",
      cz: '**Průměrné skóre za epizodu** — hlavní číslo udávající, jak dobře si agent vede.\nCelým cílem agenta je tuto křivku během tréninku tlačit nahoru, takže rostoucí odměna znamená, že se učí.',
    },
    perEnv: {
      cartpole: {
        en: 'CartPole gives +1 for every step the pole stays up, capped at 500. So reward ≈ 500 means solved — the curve should climb from roughly 20 toward 500.',
        cz: 'CartPole dává +1 za každý krok, kdy tyč zůstane vzhůru, maximálně 500. Odměna ≈ 500 tedy znamená vyřešeno — křivka by měla stoupat zhruba z 20 k 500.',
      },
      lunarlander: {
        en: 'LunarLander rewards a gentle, on-target landing: roughly +100 for landing, +10 per leg touching, small fuel penalties for firing engines, and −100 for a crash. Scores start negative and a run is "solved" at an average of 200, so the curve should climb from below zero toward 200+.',
        cz: 'LunarLander odměňuje jemné přistání na cíli: zhruba +100 za přistání, +10 za každou nohu na zemi, malé penalizace za palivo při zážehu trysek a −100 za havárii. Skóre začíná v záporu a běh je „vyřešen“ při průměru 200, takže křivka by měla stoupat z podnuly k 200+.',
      },
      mountaincar: {
        en: 'MountainCar gives −1 every step until the flag, so the best score is a small negative (around −85 to −110) and the curve climbs from −200 upward. A flat line at −200 means the agent has not reached the flag yet — that is the exploration problem, not a bug.',
        cz: 'MountainCar dává −1 každý krok až k vlajce, takže nejlepší skóre je malé záporné (kolem −85 až −110) a křivka stoupá od −200 nahoru. Rovná čára na −200 znamená, že agent ještě nedojel k vlajce — to je ten problém se zkoumáním, ne chyba.',
      },
      acrobot: {
        en: 'Acrobot also gives −1 per step until the tip swings above the bar, so scores are negative. The curve climbs from about −500 toward −100 as the agent learns to swing up faster.',
        cz: 'Acrobot také dává −1 za krok, dokud se konec nevyhoupne nad tyč, takže skóre jsou záporná. Křivka stoupá zhruba z −500 k −100, jak se agent učí vyhoupnout rychleji.',
      },
    },
  },

  loss: {
    general: {
      en: '**Training diagnostic** — roughly how much the network adjusts itself on each update.\nUnlike reward, lower is not automatically better and it does not climb steadily; it wobbles as the agent learns. Use it to spot instability (wild spikes), not as a score.',
      cz: '**Diagnostika tréninku** — zhruba jak moc se síť při každé aktualizaci upraví.\nNa rozdíl od odměny zde nižší hodnota není automaticky lepší a neroste plynule; kolísá, jak se agent učí. Slouží k odhalení nestability (divoké výkyvy), ne jako skóre.',
    },
    perEnv: {
      cartpole: {
        en: 'For CartPole the loss just bounces around small values throughout — that is normal. Judge progress from the reward curve, not this one.',
        cz: 'U CartPole se ztráta po celou dobu jen pohybuje kolem malých hodnot — to je normální. Postup posuzujte podle křivky odměny, ne podle ní.',
      },
      lunarlander: {
        en: 'As with CartPole, the loss just wobbles — judge LunarLander progress from the reward curve, not this one.',
        cz: 'Stejně jako u CartPole ztráta jen kolísá — pokrok u LunarLanderu posuzujte podle křivky odměny, ne podle ní.',
      },
      mountaincar: {
        en: 'As always, the loss just wobbles — judge MountainCar progress from the reward curve (is it lifting off −200?), not from this one.',
        cz: 'Jako vždy ztráta jen kolísá — pokrok u MountainCar posuzujte podle křivky odměny (odlepuje se od −200?), ne podle ní.',
      },
      acrobot: {
        en: 'The loss wobbles here too; read progress from the reward curve climbing toward −100, not from the loss.',
        cz: 'Ztráta i tady kolísá; pokrok čtěte z křivky odměny stoupající k −100, ne ze ztráty.',
      },
    },
  },

  fitness: {
    general: {
      en: '**The neuroevolution counterpart of reward.**\nInstead of one agent improving by gradient steps, a whole population is scored each generation and the best "genomes" are bred together. The chart shows best / average / worst across generations.',
      cz: '**Protějšek odměny u neuroevoluce.**\nMísto jednoho agenta zlepšujícího se gradientními kroky se každou generaci ohodnotí celá populace a nejlepší „genomy“ se zkříží. Graf ukazuje nejlepší / průměrnou / nejhorší hodnotu napříč generacemi.',
    },
    perEnv: {
      cartpole: {
        en: 'Same 0–500 scale as reward for CartPole. This tab comes to life once you train with the Neuroevolution algorithm (Phase C).',
        cz: 'Stejná škála 0–500 jako odměna pro CartPole. Tato záložka ožije, jakmile trénujete algoritmem Neuroevoluce (fáze C).',
      },
      lunarlander: {
        en: 'Same idea as reward, but per generation. For LunarLander fitness starts negative (crashes) and rises slowly; with the small evolution network it often plateaus below the 200 "solved" line — switch to PPO to reach a clean landing.',
        cz: 'Stejný princip jako odměna, ale za generaci. U LunarLanderu začíná fitness v záporu (havárie) a stoupá pomalu; s malou evoluční sítí často uvázne pod hranicí 200 („vyřešeno“) — pro čisté přistání přepněte na PPO.',
      },
      mountaincar: {
        en: "Same idea as reward, but per generation. Fitness often sits at −200 until some network first reaches the flag, then jumps up — neuroevolution's population search is well suited to making that discovery.",
        cz: 'Stejný princip jako odměna, ale za generaci. Fitness často vězí na −200, dokud nějaká síť poprvé nedojede k vlajce, pak vyskočí — populační hledání neuroevoluce se k takovému objevu dobře hodí.',
      },
      acrobot: {
        en: 'Same as reward, but per generation. For Acrobot fitness starts deeply negative and rises steadily toward −100 as the population learns to pump the arm up.',
        cz: 'Stejné jako odměna, ale za generaci. U Acrobotu začíná fitness hluboko v záporu a plynule stoupá k −100, jak se populace učí rameno vyhoupnout.',
      },
    },
  },

  smooth: {
    general: {
      en: '**Display-only smoothing** for the chart — it does not change training.\nThe chart draws the raw values as a faint line and overlays a bold smoothed line (an exponential moving average). Lower the slider for a calmer, more readable trend; set it to 1.0 to see every noisy data point exactly.',
      cz: '**Vyhlazení pouze pro zobrazení** grafu — trénink nijak nemění.\nGraf kreslí surové hodnoty slabou čarou a překrývá je výraznou vyhlazenou čarou (exponenciální klouzavý průměr). Snižte posuvník pro klidnější, čitelnější trend; nastavte na 1.0, abyste viděli každý zašuměný bod přesně.',
    },
    recommended: {
      en: 'Around 0.3 — smooth enough to read the trend without hiding real changes.',
      cz: 'Kolem 0.3 — dost vyhlazené pro čtení trendu, aniž by se skryly skutečné změny.',
    },
    range: '0.05 – 1.0  (1.0 = raw, no smoothing)',
  },

  // ── Top-bar chips + panel descriptions (C2) ───────────────────────────────
  // Concept popups for the header chips and the whole Evolution Stats panel.

  topbar_gen: {
    general: {
      en: '**Generation** — which breeding round neuroevolution is currently on, shown as current / total.\nEach generation scores the whole population, keeps the best, and breeds the next. PPO does not evolve a population, so this stays "—" in PPO mode.',
      cz: '**Generace** — kolikáté kolo šlechtění právě neuroevoluce zpracovává, zobrazeno jako aktuální / celkem.\nKaždá generace ohodnotí celou populaci, ponechá nejlepší a vyšlechtí další. PPO populaci nešlechtí, takže v režimu PPO zůstává „—“.',
    },
  },

  topbar_iter: {
    general: {
      en: '**Iterations** — a measure of how much work the run has done.\nFor PPO it counts completed rollout-and-update cycles; for neuroevolution it counts the total network evaluations so far (generation × population).',
      cz: '**Iterace** — měřítko, kolik práce běh odvedl.\nU PPO počítá dokončené cykly „sběr dat + úprava“; u neuroevoluce počítá celkový počet vyhodnocení sítí dosud (generace × populace).',
    },
  },

  topbar_best: {
    general: {
      en: '**All-time best** score this environment has ever reached on this machine.\nIt is saved to disk and survives restarts — distinct from the live session high. The ceiling depends on the game — each has its own "solved" score (CartPole 500, LunarLander 200).',
      cz: '**Vůbec nejlepší** skóre, jakého kdy toto prostředí na tomto počítači dosáhlo.\nUkládá se na disk a přežije restart — na rozdíl od nejlepšího skóre aktuální relace. Strop závisí na hře — každá má vlastní skóre „vyřešeno“ (CartPole 500, LunarLander 200).',
    },
  },

  topbar_pop: {
    general: {
      en: '**Population** — how many neural networks compete in each neuroevolution generation.\nA bigger population explores more strategies per generation but is slower to score. PPO trains a single network, so this stays "—" in PPO mode.',
      cz: '**Populace** — kolik neuronových sítí soutěží v každé generaci neuroevoluce.\nVětší populace zkouší více strategií za generaci, ale ohodnocení je pomalejší. PPO trénuje jedinou síť, takže v režimu PPO zůstává „—“.',
    },
  },

  evolution_stats: {
    general: {
      en: "This panel summarises the current neuroevolution generation.\n**Generation** — the breeding round (current / total).\n**Total Iters** — how many networks have been scored so far (generation × population).\n**Best / Avg / Worst** — the fitness of the top, mean and bottom network this generation. Fitness = the average reward over the evaluation episodes (its scale depends on the game; reaching the game's solved score = mastered); a healthy run pushes Best up and Avg follows.\n**Mutation spread** — a histogram of the random weight tweaks used to breed this generation's offspring: centred on zero, bell-shaped, and wider when the Mutation Rate is higher.\nWatch Best climb generation by generation — once it nears the ceiling, the population has mastered the task.",
      cz: 'Tento panel shrnuje aktuální generaci neuroevoluce.\n**Generace** — kolo šlechtění (aktuální / celkem).\n**Celkem iterací** — kolik sítí už bylo ohodnoceno (generace × populace).\n**Nejlepší / Průměr / Nejhorší** — fitness nejlepší, průměrné a nejhorší sítě v této generaci. Fitness = průměrná odměna za vyhodnocovací epizody (škála závisí na hře; dosažení skóre „vyřešeno“ = zvládnuto); zdravý běh tlačí Nejlepší nahoru a Průměr ho následuje.\n**Rozptyl mutací** — histogram náhodných úprav vah použitých k vyšlechtění potomků této generace: vystředěný na nule, tvarem jako zvon, širší při vyšší Míře mutace.\nSleduj, jak Nejlepší stoupá generaci za generací — jakmile se přiblíží stropu, populace úlohu zvládla.',
    },
  },
}
