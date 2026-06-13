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
      en: 'PPO — the reliable, general-purpose default that also scales to hard games later. On an easy task like CartPole, though, Neuroevolution often reaches 500 faster in wall-clock time — try both.',
      cz: 'PPO — spolehlivá, univerzální volba, která později zvládne i těžké hry. U snadné úlohy jako CartPole ale Neuroevoluce často dosáhne 500 rychleji v reálném čase — vyzkoušejte oba.',
    },
    perEnv: {
      cartpole: {
        en: 'Both solve CartPole. Counter-intuitively, Neuroevolution is often the faster of the two here — within a few wall-clock seconds — because the network is tiny and there are no gradient updates, just scoring a population. PPO is steadier and is the method that scales to hard problems, but on this easy task it usually needs more wall-clock time (it collects rollouts and runs gradient steps).',
        cz: 'Oba CartPole vyřeší. Možná překvapivě bývá tady rychlejší Neuroevoluce — během pár vteřin reálného času — protože síť je drobná a nejsou žádné gradientní úpravy, jen ohodnocení populace. PPO je stabilnější a je to metoda, která zvládne i těžké úlohy, ale u téhle snadné úlohy obvykle potřebuje víc reálného času (sbírá data a počítá gradientní kroky).',
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
    },
  },

  ent_coef: {
    general: {
      en: '**Entropy bonus** — rewards the agent for staying a bit random and exploring.\nRaise it if the agent settles too early on a mediocre habit. Keep it near 0 when the task is simple enough to solve by exploiting what it learns.',
      cz: '**Bonus za entropii** — odměňuje agenta za to, že zůstane trochu náhodný a zkoumá.\nZvyšte ho, pokud agent příliš brzy zapadne do průměrného návyku. Nechte ho u 0, když je úloha dost jednoduchá na vyřešení tím, co se naučil.',
    },
    recommended: {
      en: '0.0 — CartPole needs no extra exploration push.',
      cz: '0.0 — CartPole nepotřebuje žádný extra tlak na zkoumání.',
    },
    range: '0.0 – 0.05',
    perEnv: {
      cartpole: {
        en: "CartPole's two actions are easy to explore, so 0 is fine. A tiny value (0.01) rarely hurts if learning stalls.",
        cz: 'Dvě akce CartPole se zkoumají snadno, takže 0 stačí. Drobná hodnota (0.01) málokdy uškodí, pokud učení uvázne.',
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
    },
  },

  total_steps: {
    general: {
      en: '**Training budget** — how many environment steps the agent gets to learn from before stopping.\nMore steps mean more practice and usually a better policy, up to the point where it has mastered the task.',
      cz: '**Tréninkový rozpočet** — kolik kroků v prostředí agent dostane na učení, než se zastaví.\nVíce kroků znamená více cviku a obvykle lepší strategii, dokud úlohu nezvládne.',
    },
    recommended: {
      en: '50k — comfortably solves CartPole on CPU in well under a minute.',
      cz: '50k — pohodlně vyřeší CartPole na CPU výrazně pod minutu.',
    },
    range: '10k – 200k',
    perEnv: {
      cartpole: {
        en: 'CartPole is usually solved by ~30–50k steps. Beyond that the reward just sits at 500 — more steps won\'t help.',
        cz: 'CartPole se obvykle vyřeší kolem 30–50k kroků. Dál už odměna jen sedí na 500 — více kroků nepomůže.',
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
    },
  },

  generations: {
    general: {
      en: '**Breeding rounds** — the evolution equivalent of training length.\nEach generation scores the whole population, keeps the best, and breeds the next. More generations give more chances to refine, up to the point the task is mastered.',
      cz: '**Kola šlechtění** — evoluční obdoba délky tréninku.\nKaždá generace ohodnotí celou populaci, ponechá nejlepší a vyšlechtí další. Více generací dává více příležitostí k vylepšení, dokud úloha není zvládnuta.',
    },
    recommended: {
      en: '30 — comfortably enough for the herd to reach a near-perfect CartPole score.',
      cz: '30 — pohodlně dost na to, aby stádo dosáhlo téměř dokonalého skóre v CartPole.',
    },
    range: '5 – 200',
    perEnv: {
      cartpole: {
        en: 'CartPole is often essentially solved within 10–20 generations; beyond that the best fitness just sits near 500, much like extra steps do for PPO.',
        cz: 'CartPole bývá v podstatě vyřešen během 10–20 generací; dále už nejlepší fitness jen sedí poblíž 500, podobně jako kroky navíc u PPO.',
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
      en: '**All-time best** score this environment has ever reached on this machine.\nIt is saved to disk and survives restarts — distinct from the live session high. For CartPole the ceiling is 500 (a pole balanced for the whole episode).',
      cz: '**Vůbec nejlepší** skóre, jakého kdy toto prostředí na tomto počítači dosáhlo.\nUkládá se na disk a přežije restart — na rozdíl od nejlepšího skóre aktuální relace. U CartPole je strop 500 (tyč udržená po celou epizodu).',
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
      en: "This panel summarises the current neuroevolution generation.\n**Generation** — the breeding round (current / total).\n**Total Iters** — how many networks have been scored so far (generation × population).\n**Best / Avg / Worst** — the fitness of the top, mean and bottom network this generation. Fitness = the average reward over the evaluation episodes (0–500 for CartPole, ~500 = solved); a healthy run pushes Best up and Avg follows.\n**Mutation spread** — a histogram of the random weight tweaks used to breed this generation's offspring: centred on zero, bell-shaped, and wider when the Mutation Rate is higher.\nWatch Best climb generation by generation — once it nears the ceiling, the population has mastered the task.",
      cz: 'Tento panel shrnuje aktuální generaci neuroevoluce.\n**Generace** — kolo šlechtění (aktuální / celkem).\n**Celkem iterací** — kolik sítí už bylo ohodnoceno (generace × populace).\n**Nejlepší / Průměr / Nejhorší** — fitness nejlepší, průměrné a nejhorší sítě v této generaci. Fitness = průměrná odměna za vyhodnocovací epizody (0–500 pro CartPole, ~500 = vyřešeno); zdravý běh tlačí Nejlepší nahoru a Průměr ho následuje.\n**Rozptyl mutací** — histogram náhodných úprav vah použitých k vyšlechtění potomků této generace: vystředěný na nule, tvarem jako zvon, širší při vyšší Míře mutace.\nSleduj, jak Nejlepší stoupá generaci za generací — jakmile se přiblíží stropu, populace úlohu zvládla.',
    },
  },
}
