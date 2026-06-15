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
      en: '**How the agent learns.**\n**PPO** (reinforcement learning) — tweaks one neural network with gradients after each batch of play; steady and sample-efficient.\n**Neuroevolution** — keeps a whole population of networks, scores them, and breeds the best (mutation + crossover) each generation; simple, gradient-free, like "survival of the fittest".\n**Q-learning** — builds a plain table of "how good is each action in each state" and refines it from experience; the classic value-based method, available on the small grid-world games where you can literally watch the table fill in.',
      cz: '**Jak se agent učí.**\n**PPO** (zpětnovazební učení) — upravuje jednu neuronovou síť pomocí gradientů po každé dávce hraní; stabilní a úsporné na data.\n**Neuroevoluce** — udržuje celou populaci sítí, ohodnotí je a v každé generaci množí ty nejlepší (mutace + křížení); jednoduchá, bez gradientů, jako „přežití nejschopnějších“.\n**Q-učení** — sestavuje jednoduchou tabulku „jak dobrá je každá akce v každém stavu“ a vylepšuje ji ze zkušenosti; klasická hodnotová metoda, dostupná u malých mřížkových her, kde můžete doslova sledovat, jak se tabulka plní.',
    },
    recommended: {
      en: 'PPO — the reliable, general-purpose default that also scales to harder games. Neuroevolution is gradient-free and can be surprisingly fast on simple tasks. On the grid-world games, Q-learning is the star: it solves them cleanly and you can watch its table fill in. Try all three and compare (see the per-game note below).',
      cz: 'PPO — spolehlivá, univerzální volba, která zvládne i těžší hry. Neuroevoluce je bezgradientní a u jednoduchých úloh bývá překvapivě rychlá. U mřížkových her je hvězdou Q-učení: vyřeší je čistě a můžete sledovat, jak se jeho tabulka plní. Vyzkoušejte všechny tři a porovnejte (viz poznámka k dané hře níže).',
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
      pendulum: {
        en: 'Both work on Pendulum. PPO is the natural fit for smooth continuous control and learns a clean swing-up-and-hold. Neuroevolution also works — its networks output a continuous torque — and is worth comparing, though PPO is usually the steadier of the two here.',
        cz: 'Na Pendulu fungují obě. PPO je přirozená volba pro plynulé spojité řízení a naučí se čisté vyhoupnutí a udržení. Neuroevoluce také funguje — její sítě vydávají spojitý točivý moment — a stojí za porovnání, i když PPO tu bývá stabilnější.',
      },
      mountaincarcontinuous: {
        en: 'A continuous exploration trap. PPO often stalls near 0: it rarely stumbles onto the flag, and the small force penalty discourages trying. Neuroevolution tends to do better — among a whole population some network reaches the flag and that success spreads. A great task for comparing the two.',
        cz: 'Spojitá past na zkoumání. PPO často uvázne poblíž 0: na vlajku jen zřídka náhodou narazí a malá penalizace za sílu odrazuje od zkoušení. Neuroevoluce tu bývá lepší — v celé populaci některá síť dojede k vlajce a úspěch se rozšíří. Skvělá úloha pro porovnání obou.',
      },
      bipedalwalker: {
        en: 'Only PPO is offered here. Walking is a hard continuous-control task with four leg joints, and PPO is the method that actually learns a gait (given a few million steps). Neuroevolution is deliberately turned off — population search is impractical on this. Training is reserved for a GPU machine, so Run is disabled here; you can still play it by hand now and watch a trained AI once one exists.',
        cz: 'Tady je k dispozici jen PPO. Chůze je těžká úloha spojitého řízení se čtyřmi klouby nohou a PPO je metoda, která chůzi skutečně zvládne (za pár milionů kroků). Neuroevoluce je záměrně vypnutá — populační hledání je tu nepraktické. Trénink je vyhrazen pro stroj s GPU, takže Spustit je zde zakázané; rukama si to ale zahrajete už teď a natrénovanou AI uvidíte, jakmile bude existovat.',
      },
      bipedalwalkerhardcore: {
        en: 'Only PPO, same as the standard course — and the hardcore terrain (ladders, stumps, pits) makes it one of the hardest continuous-control benchmarks, needing the largest budget. Training is GPU-only (Run disabled here); play it by hand now.',
        cz: 'Jen PPO, stejně jako u standardní dráhy — a hardcore terén (žebříky, pařezy, jámy) z ní dělá jeden z nejtěžších benchmarků spojitého řízení, který potřebuje největší rozpočet. Trénink je jen na GPU (Spustit je zde zakázané); rukama si to zahrajete už teď.',
      },
      frozenlake: {
        en: 'Both can solve it. The state is a single grid cell (one-hot–encoded for the network), and the reward is sparse — only reaching the goal pays — so a little exploration matters. Because the ice slips, success is a *rate*: even a good policy scores below 1.0. A nice task to compare PPO and Neuroevolution.',
        cz: 'Vyřešit ho zvládnou obě. Stav je jediná buňka mřížky (pro síť kódovaná jako one-hot) a odměna je řídká — platí jen dosažení cíle — takže záleží na trošce zkoumání. Protože led klouže, je úspěch *míra*: i dobrá strategie skóruje pod 1,0. Pěkná úloha pro porovnání PPO a Neuroevoluce.',
      },
      frozenlake_noslip: {
        en: 'The easiest grid — both solve it fast. With no slipping it is a clean shortest-path maze: Neuroevolution often nails it in a few generations, PPO in a few thousand steps. Ideal for watching the two learn side by side.',
        cz: 'Nejjednodušší mřížka — obě ji vyřeší rychle. Bez klouzání jde o čisté bludiště s nejkratší cestou: Neuroevoluce ji často zvládne za pár generací, PPO za pár tisíc kroků. Ideální pro pozorování obou vedle sebe.',
      },
      frozenlake8x8: {
        en: 'Both can learn it, but it is much harder: 64 cells and a long slippery path mean success is rare early on. Give either method a generous budget — PPO with a little Entropy, or Neuroevolution with a larger population, both help.',
        cz: 'Naučit se ho zvládnou obě, ale je mnohem těžší: 64 buněk a dlouhá kluzká cesta znamenají, že úspěch je zpočátku vzácný. Dejte oběma štědrý rozpočet — pomůže PPO s trochou entropie i Neuroevoluce s větší populací.',
      },
      taxi: {
        en: 'Taxi has 500 states and a long pick-up-then-drop-off routine, which is hard for both methods here: PPO needs a lot of practice and the tiny evolution network struggles with so many states (you will often see the agent wander without ever picking the passenger up). This is exactly the kind of small, fully-observed task where a value-based method like tabular Q-learning shines.',
        cz: 'Taxi má 500 stavů a dlouhou rutinu „naber a vysaď", což je tu pro obě metody těžké: PPO potřebuje hodně tréninku a drobná evoluční síť s tolika stavy bojuje (často uvidíte agenta bloudit, aniž by cestujícího vůbec nabral). Přesně na takových malých, plně pozorovatelných úlohách vyniká hodnotová metoda jako tabulkové Q-učení.',
      },
      cliffwalking: {
        en: 'CliffWalking is a classic exploration trap, not a fair fight for these two. Sitting safely in the corner scores about −200 (just running out the clock), while heading for the goal means risking the −100 cliff falls first — so the reward landscape lures PPO and neuroevolution into the "play it safe, never reach the goal" corner and tends to keep them there (≈ −200, ~0% on the meter). PPO occasionally breaks out, but it is unreliable and seed-dependent, and more steps usually will not rescue a run that has already settled — this is not a bug, it is the well-known weakness of policy-gradient and population methods on a deceptive-reward grid. It is exactly the textbook task that value-based tabular Q-learning solves cleanly.',
        cz: 'CliffWalking je klasická past na zkoumání, ne férový souboj pro tyhle dvě metody. Bezpečné sezení v rohu dá kolem −200 (jen vyčerpá čas), kdežto cesta k cíli nejdřív znamená riskovat pády z útesu za −100 — odměnová krajina tak láká PPO i neuroevoluci do rohu „hraj na jistotu, do cíle nikdy" a většinou je tam drží (≈ −200, ~0 % na ukazateli). PPO se občas vymaní, ale nespolehlivě a podle seedu, a víc kroků už zaseknutý běh většinou nezachrání — to není chyba, je to známá slabina metod policy-gradient a populačních metod na mřížce s klamavou odměnou. Je to přesně ta učebnicová úloha, kterou hodnotové tabulkové Q-učení vyřeší čistě.',
      },
      minigrid_empty: {
        en: 'Both can be tried, and Empty-5x5 is easy enough for either. PPO is the clean choice — it reaches ~0.95 in seconds. Neuroevolution can also find the goal, but each network carries a huge ~2835-input layer (the flattened 7×7 view + the mission text), so its tiny genome is slower and clumsier than on a small Toy Text grid.',
        cz: 'Vyzkoušet lze obě a Empty-5x5 je dost snadné pro obě. Čistou volbou je PPO — dosáhne ~0,95 během vteřin. Neuroevoluce cíl také najde, ale každá síť nese obří vstupní vrstvu ~2835 (zploštělý pohled 7×7 + text mise), takže její drobný genom je pomalejší a neohrabanější než na malé mřížce Toy Text.',
      },
      minigrid_fourrooms: {
        en: 'PPO is the practical choice. Sparse reward plus a randomly-placed start and goal makes this a real exploration problem — PPO can solve it, and nudging Entropy up from its 0 default (see the Entropy note) helps it find the goal. Neuroevolution runs, but the ~2835-dim observation makes its small-network genome weak, so it explores poorly here — a fair comparison, not a like-for-like contest.',
        cz: 'Praktickou volbou je PPO. Řídká odměna spolu s náhodně umístěným startem i cílem dělá z této úlohy skutečný problém zkoumání — PPO ho vyřešit umí a pomůže mu, když entropii zvýšíte z výchozí 0 (viz poznámka u Entropie). Neuroevoluce běží, ale obří obs (~2835) dělá její malou síť slabou, takže tu zkoumá špatně — férové srovnání, ne souboj rovného s rovným.',
      },
      minigrid_doorkey: {
        en: 'PPO is the right tool. The key → door → goal sequence needs credit assignment across many steps, which PPO handles; the tiny evolution network struggles both with the big observation and the multi-step sub-goal. Watch PPO learn the ordering over a few hundred thousand steps.',
        cz: 'Správným nástrojem je PPO. Posloupnost klíč → dveře → cíl vyžaduje přiřazení zásluh přes mnoho kroků, což PPO zvládá; drobná evoluční síť bojuje jak s velkým obs, tak s vícekrokovým dílčím cílem. Sledujte, jak se PPO naučí pořadí během pár set tisíc kroků.',
      },
      minigrid_keycorridor: {
        en: 'Realistically PPO only. This hierarchical task (find the key behind a door, then unlock the ball\'s room) is hard even for PPO and needs a large budget; neuroevolution\'s small genome on a 2835-dim obs rarely makes progress — it is here for comparison, but expect it to stay near 0.',
        cz: 'Realisticky jen PPO. Tato hierarchická úloha (najít klíč za dveřmi a pak odemknout místnost s míčkem) je těžká i pro PPO a potřebuje velký rozpočet; malý genom neuroevoluce na obs o 2835 rozměrech jen zřídka udělá pokrok — je tu pro srovnání, ale čekejte, že zůstane poblíž 0.',
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
      pendulum: {
        en: 'The default 3e-4 is a solid starting point for this smooth task. Pendulum is stable to learn, so give it more steps rather than a higher rate, which can make the swing-up jittery.',
        cz: 'Výchozí 3e-4 je pro tuto plynulou úlohu dobrý výchozí bod. Pendulum se učí stabilně, takže mu dejte raději víc kroků než vyšší rychlost, která může vyhoupnutí rozkmitat.',
      },
      mountaincarcontinuous: {
        en: 'The default 3e-4 is fine. As with MountainCar the bottleneck is exploration, not step size — if the score sits near 0, raise the Entropy bonus or try Neuroevolution before touching the rate.',
        cz: 'Výchozí 3e-4 je v pořádku. Stejně jako u MountainCar je překážkou zkoumání, ne velikost kroku — pokud skóre vězí poblíž 0, zvyšte raději bonus za entropii nebo zkuste Neuroevoluci než měnit rychlost.',
      },
      bipedalwalker: {
        en: 'The default 3e-4 is the standard starting point for BipedalWalker too. It is a hard, sometimes unstable task, so if anything lower it rather than raise it — too large a step can make the gait collapse. (Training is GPU-only, so this applies once you train on the desktop.)',
        cz: 'Výchozí 3e-4 je standardní výchozí bod i pro BipedalWalker. Je to těžká, místy nestabilní úloha, takže ji spíš snižte než zvyšujte — příliš velký krok může chůzi rozhodit. (Trénink je jen na GPU, takže se to týká až tréninku na desktopu.)',
      },
      bipedalwalkerhardcore: {
        en: 'Same 3e-4 starting point; the harder terrain is even more prone to instability, so keep the rate moderate and lean on more steps instead.',
        cz: 'Stejný výchozí bod 3e-4; těžší terén je ještě náchylnější k nestabilitě, takže držte rychlost umírněnou a spoléhejte raději na víc kroků.',
      },
      frozenlake: {
        en: 'Default 3e-4 is fine. The bottleneck here is exploration (slippery ice, sparse reward), not step size — if it stalls, raise Entropy or add steps before touching this.',
        cz: 'Výchozí 3e-4 je v pořádku. Překážkou je tu zkoumání (kluzký led, řídká odměna), ne velikost kroku — pokud učení vázne, zvyšte raději entropii nebo přidejte kroky.',
      },
      frozenlake_noslip: {
        en: '3e-4 solves this easy deterministic maze quickly; there is little reason to change it.',
        cz: '3e-4 toto snadné deterministické bludiště vyřeší rychle; není důvod ji měnit.',
      },
      frozenlake8x8: {
        en: '3e-4 works on the bigger lake too — give it more steps rather than a larger step size, which can destabilise the longer task.',
        cz: '3e-4 funguje i na větším jezeře — dejte mu raději víc kroků než větší krok, který může delší úlohu rozhodit.',
      },
      taxi: {
        en: '3e-4 is a solid default for Taxi. With 500 states, more steps help far more than a higher learning rate.',
        cz: '3e-4 je pro Taxi dobrá výchozí hodnota. Při 500 stavech pomůže víc kroků mnohem víc než vyšší rychlost učení.',
      },
      cliffwalking: {
        en: '3e-4 is fine. The −100 cliff penalties are large, so a much higher rate can make learning lurch — keep it moderate.',
        cz: '3e-4 je v pořádku. Penalizace −100 za útes jsou velké, takže výrazně vyšší rychlost může učení rozkolísat — držte ji umírněnou.',
      },
      minigrid_empty: {
        en: '3e-4 solves Empty-5x5 quickly; there is no reason to change it.',
        cz: '3e-4 vyřeší Empty-5x5 rychle; není důvod ji měnit.',
      },
      minigrid_fourrooms: {
        en: '3e-4 is fine. The bottleneck on MiniGrid is exploration, not step size — add steps or raise Entropy before touching this.',
        cz: '3e-4 je v pořádku. Překážkou u MiniGridu je zkoumání, ne velikost kroku — přidejte raději kroky nebo zvyšte entropii.',
      },
      minigrid_doorkey: {
        en: '3e-4 works for the key-and-door task; give it more steps rather than a bigger rate, which can destabilise the multi-step learning.',
        cz: '3e-4 pro úlohu s klíčem a dveřmi funguje; dejte mu raději víc kroků než větší krok, který může vícekrokové učení rozhodit.',
      },
      minigrid_keycorridor: {
        en: '3e-4 is fine — this hard task needs budget and exploration far more than a higher learning rate.',
        cz: '3e-4 je v pořádku — tato těžká úloha potřebuje rozpočet a zkoumání mnohem víc než vyšší rychlost učení.',
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
      pendulum: {
        en: 'A swing-up followed by a long balancing hold rewards planning ahead, so keep γ high (0.99). Too low and the agent will not value staying upright over the next few steps\' effort.',
        cz: 'Vyhoupnutí a následné dlouhé udržení rovnováhy odměňuje plánování dopředu, takže nechte γ vysoké (0.99). Při nízkém γ agent nedocení setrvání ve svislé poloze oproti námaze nejbližších kroků.',
      },
      mountaincarcontinuous: {
        en: 'Reaching the flag is a long build-up, so keep γ high (0.99): the agent must value the distant +100 over the small force cost it pays each step.',
        cz: 'Dojezd k vlajce je dlouhé rozhoupávání, takže nechte γ vysoké (0.99): agent musí cenit vzdálených +100 víc než malou cenu za sílu, kterou platí každý krok.',
      },
      bipedalwalker: {
        en: 'Walking forward is a long sequence of coordinated steps, so keep γ high (0.99): the agent must value staying upright and making progress many steps ahead, not just the next torque.',
        cz: 'Chůze vpřed je dlouhá řada koordinovaných kroků, takže nechte γ vysoké (0.99): agent musí cenit setrvání ve vzpřímené poloze a postup mnoho kroků dopředu, ne jen další moment.',
      },
      bipedalwalkerhardcore: {
        en: 'Keep γ high (0.99) — clearing each obstacle pays off only over the steps that follow, so the agent must plan well ahead.',
        cz: 'Nechte γ vysoké (0.99) — překonání každé překážky se vyplatí až v dalších krocích, takže agent musí plánovat hodně dopředu.',
      },
      frozenlake: {
        en: 'Reaching the goal is a long sequence with no reward along the way, so keep γ high (0.99) — the agent must value the distant goal across many zero-reward steps. Too low and it cannot "see" the goal from far away.',
        cz: 'Dosažení cíle je dlouhá řada kroků bez odměny po cestě, takže nechte γ vysoké (0.99) — agent musí cenit vzdálený cíl přes mnoho kroků s nulovou odměnou. Při nízkém γ cíl zdálky „neuvidí“.',
      },
      frozenlake_noslip: {
        en: 'Keep γ high (0.99): even on this easy map the only reward is at the goal, so the agent must value it across the whole path.',
        cz: 'Nechte γ vysoké (0.99): i na této snadné mapě je jediná odměna v cíli, takže ji agent musí cenit přes celou cestu.',
      },
      frozenlake8x8: {
        en: 'Even more important on the bigger board — the goal is many steps away, so a high γ (0.99) is essential to carry its value back across the lake.',
        cz: 'Na větší desce ještě důležitější — cíl je mnoho kroků daleko, takže vysoké γ (0.99) je nutné, aby se jeho hodnota přenesla zpět přes jezero.',
      },
      taxi: {
        en: 'Keep γ high (0.99): the +20 drop-off comes only after a long run of −1 steps, so the agent must value that distant payoff to plan the whole trip.',
        cz: 'Nechte γ vysoké (0.99): +20 za vysazení přijde až po dlouhé řadě kroků po −1, takže agent musí tuto vzdálenou výplatu cenit, aby naplánoval celou jízdu.',
      },
      cliffwalking: {
        en: 'Keep γ high (0.99) so the goal\'s value reaches back across the path — it also makes the −100 cliff loom large enough to plan around.',
        cz: 'Nechte γ vysoké (0.99), aby se hodnota cíle přenesla zpět přes cestu — díky tomu také −100 za útes „čouhá“ dost na to, aby se mu agent vyhýbal.',
      },
      minigrid_empty: {
        en: 'Keep γ high (0.99) so the single goal reward propagates back the few steps to the start.',
        cz: 'Nechte γ vysoké (0.99), aby se jediná odměna za cíl přenesla zpět těch pár kroků ke startu.',
      },
      minigrid_fourrooms: {
        en: 'A high γ (0.99) is essential — the goal can be rooms away, so its value must travel far back through the doorways.',
        cz: 'Vysoké γ (0.99) je nutné — cíl může být místnosti daleko, takže jeho hodnota musí cestovat daleko zpět přes průchody.',
      },
      minigrid_doorkey: {
        en: 'Keep γ high (0.99): the goal reward must reach back past the door and the key pickup for the agent to value those earlier sub-steps.',
        cz: 'Nechte γ vysoké (0.99): odměna za cíl musí dosáhnout zpět za dveře a sebrání klíče, aby agent cenil tyto dřívější dílčí kroky.',
      },
      minigrid_keycorridor: {
        en: 'A high γ (0.99) is critical — the ball reward is many steps (key, door, room) away and must propagate all the way back.',
        cz: 'Vysoké γ (0.99) je klíčové — odměna za míček je mnoho kroků daleko (klíč, dveře, místnost) a musí se přenést celou cestu zpět.',
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
      pendulum: {
        en: '0.2 works well on Pendulum; there is rarely a reason to change it.',
        cz: '0.2 na Pendulu funguje dobře; jen zřídka je důvod ji měnit.',
      },
      mountaincarcontinuous: {
        en: '0.2 is fine — clipping is not the bottleneck on MountainCarContinuous; exploration is. Leave it and adjust Entropy and steps instead.',
        cz: '0.2 je v pořádku — ořezávání není u MountainCarContinuous překážkou; tou je zkoumání. Nechte ji a laďte raději entropii a počet kroků.',
      },
      bipedalwalker: {
        en: '0.2 is the right default. BipedalWalker can be unstable to train, and the clip is what keeps one bad batch from wrecking a working gait — leave it at 0.2 and tune steps and learning rate first.',
        cz: '0.2 je správná výchozí hodnota. BipedalWalker se může trénovat nestabilně a ořezávání je to, co brání jedné špatné dávce zničit funkční chůzi — nechte ji na 0.2 a laďte nejdřív počet kroků a rychlost učení.',
      },
      bipedalwalkerhardcore: {
        en: 'Leave it at 0.2 — on this harder course the safety rail matters even more; tune budget and exploration instead.',
        cz: 'Nechte ji na 0.2 — na této těžší dráze záleží na bezpečnostním zábradlí ještě víc; laďte raději rozpočet a zkoumání.',
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
      pendulum: {
        en: 'Pendulum is learnable at 0, and PPO\'s continuous action head already explores with its own noise. A small value (~0.01) can still speed the early search for the swing-up.',
        cz: 'Pendulum je řešitelný i při 0 a spojitá akční hlava PPO už zkoumá vlastním šumem. Drobná hodnota (~0,01) přesto může urychlit počáteční hledání vyhoupnutí.',
      },
      mountaincarcontinuous: {
        en: 'This is the key dial for MountainCarContinuous. The reward is sparse, so a little extra exploration is often what lets PPO find the flag at all — raise it (e.g. 0.01–0.1) if the score is stuck near 0, or try Neuroevolution.',
        cz: 'U MountainCarContinuous je tohle klíčový knoflík. Odměna je řídká, takže trocha zkoumání navíc je často to, co PPO vůbec umožní najít vlajku — zvyšte ji (např. 0,01–0,1), pokud skóre vězí poblíž 0, nebo zkuste Neuroevoluci.',
      },
      bipedalwalker: {
        en: 'PPO\'s continuous action head already explores with its own noise, so 0 can work, but a small bonus (e.g. 0.01) often helps it discover a forward gait instead of standing still. Do not push it too high — too much randomness just makes the robot flail and fall.',
        cz: 'Spojitá akční hlava PPO už zkoumá vlastním šumem, takže 0 může stačit, ale drobný bonus (např. 0,01) často pomůže objevit chůzi vpřed místo stání na místě. Nezvyšujte ho příliš — moc náhodnosti robota jen rozhází a shodí.',
      },
      bipedalwalkerhardcore: {
        en: 'A small bonus (e.g. 0.01) helps it explore the moves needed to get over obstacles, but keep it modest — too much and it flails into a fall.',
        cz: 'Drobný bonus (např. 0,01) pomůže prozkoumat pohyby potřebné k překonání překážek, ale držte ho mírný — při přemíře robot spadne.',
      },
      frozenlake: {
        en: 'Exploration matters here — only the goal pays, so a small bonus (e.g. 0.01) helps PPO discover a route before settling. At 0 it can stall on the slippery map, never finding the goal.',
        cz: 'Zkoumání tu hraje roli — platí jen cíl, takže drobný bonus (např. 0,01) pomůže PPO objevit cestu, než se ustálí. Při 0 může na kluzké mapě uváznout a cíl nikdy nenajít.',
      },
      frozenlake_noslip: {
        en: '0 is usually fine on this easy deterministic map; a tiny 0.01 will not hurt if learning stalls.',
        cz: '0 na této snadné deterministické mapě obvykle stačí; drobných 0,01 neuškodí, pokud učení vázne.',
      },
      frozenlake8x8: {
        en: 'The key dial here — with 64 cells and sparse reward, raise it (e.g. 0.01–0.05) so PPO explores enough to ever reach the goal across the big lake.',
        cz: 'Tady klíčový knoflík — při 64 buňkách a řídké odměně ji zvyšte (např. 0,01–0,05), aby PPO zkoumalo dost na to, aby přes velké jezero cíl vůbec našlo.',
      },
      taxi: {
        en: 'A little exploration (e.g. 0.01) helps PPO discover the pick-up and drop-off actions early; 0 still works but can learn slowly.',
        cz: 'Trocha zkoumání (např. 0,01) pomůže PPO brzy objevit akce nabrání a vysazení; 0 také funguje, ale učení může být pomalé.',
      },
      cliffwalking: {
        en: 'A small bonus (~0.01) speeds early route-finding, but the dense −1 reward already gives a clear signal, so 0 works fine here too.',
        cz: 'Drobný bonus (~0,01) urychlí počáteční hledání cesty, ale hustá odměna −1 už dává jasný signál, takže 0 tu funguje také dobře.',
      },
      minigrid_empty: {
        en: '0 usually works on this small room (the goal is close), but a tiny 0.01 can speed up finding it. Not a critical dial here.',
        cz: 'Na této malé místnosti 0 obvykle stačí (cíl je blízko), ale drobných 0,01 může urychlit jeho nalezení. Tady to není kritický knoflík.',
      },
      minigrid_fourrooms: {
        en: 'A key dial here — raise it (e.g. 0.01–0.05) so PPO explores enough to find the randomly-placed goal across the rooms. At 0 it can wander and never reach it.',
        cz: 'Tady klíčový knoflík — zvyšte ho (např. 0,01–0,05), aby PPO zkoumalo dost na nalezení náhodně umístěného cíle napříč místnostmi. Při 0 může bloudit a cíl nikdy nedosáhnout.',
      },
      minigrid_doorkey: {
        en: 'Raise it (e.g. 0.01–0.05) so PPO tries picking up the key and opening the door before settling — with sparse reward and a sub-goal, exploration is what unlocks progress.',
        cz: 'Zvyšte ho (např. 0,01–0,05), aby PPO zkusilo sebrat klíč a otevřít dveře dřív, než se ustálí — při řídké odměně a dílčím cíli je právě zkoumání tím, co odemkne pokrok.',
      },
      minigrid_keycorridor: {
        en: 'The most important dial here — almost nothing is rewarded until the ball is picked up, so push exploration (0.01–0.1). Even then this hierarchical task is hard.',
        cz: 'Nejdůležitější knoflík tady — dokud se nesebere míček, neodměňuje se téměř nic, takže přidejte zkoumání (0,01–0,1). I tak je tato hierarchická úloha těžká.',
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
      pendulum: {
        en: 'The 3-number observation is tiny, so 2 layers is ample for Pendulum. Extra depth just slows things down.',
        cz: 'Pozorování o třech číslech je drobné, takže 2 vrstvy pro Pendulum bohatě stačí. Hloubka navíc jen vše zpomalí.',
      },
      mountaincarcontinuous: {
        en: 'The 2-number observation is tiny, so 2 layers is plenty. Extra depth will not fix MountainCarContinuous — exploration will.',
        cz: 'Pozorování o dvou číslech je drobné, takže 2 vrstvy bohatě stačí. Hloubka navíc MountainCarContinuous nevyřeší — vyřeší ho zkoumání.',
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
      pendulum: {
        en: "64 neurons easily handle Pendulum's small state. Going wider gives no benefit and just uses more CPU.",
        cz: '64 neuronů drobný stav Pendula hravě zvládne. Více nepřinese žádnou výhodu, jen spotřebuje více CPU.',
      },
      mountaincarcontinuous: {
        en: "64 neurons easily cover MountainCarContinuous's tiny state. Going wider just costs CPU without helping.",
        cz: '64 neuronů drobný stav MountainCarContinuous hravě pokryje. Širší síť jen spotřebuje CPU, aniž by pomohla.',
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
      pendulum: {
        en: 'tanh is a safe default for Pendulum; relu works too but makes little difference on such a small net.',
        cz: 'tanh je pro Pendulum bezpečná výchozí volba; relu také funguje, ale na tak malé síti je rozdíl malý.',
      },
      mountaincarcontinuous: {
        en: 'tanh is a safe default. The activation is not the lever on MountainCarContinuous — exploration is.',
        cz: 'tanh je bezpečná výchozí volba. Aktivační funkce není u MountainCarContinuous ta páka — tou je zkoumání.',
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
      pendulum: {
        en: 'Pendulum starts at a random angle and speed, so luck affects the early swing-up. Fix it for fair comparisons; change it to see how robust a policy is to the starting state.',
        cz: 'Pendulum začíná v náhodném úhlu a rychlosti, takže náhoda ovlivní počáteční vyhoupnutí. Pro férové porovnání ho zafixujte; změnou uvidíte, jak robustní je strategie vůči počátečnímu stavu.',
      },
      mountaincarcontinuous: {
        en: 'Luck matters a lot here — on some seeds a run stumbles onto the flag early and takes off, on others it never does. Fix it for fair comparisons; change it to feel how much exploration depends on chance.',
        cz: 'Náhoda tu hraje velkou roli — na některých semenech běh brzy narazí na vlajku a rozjede se, na jiných nikdy. Pro férové porovnání ho zafixujte; změnou ucítíte, jak moc zkoumání závisí na štěstí.',
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
      pendulum: {
        en: 'Pendulum usually learns a good swing-up-and-hold within a few hundred thousand steps, so the 200k ★ is a reasonable budget. Give it more if the score is not yet climbing toward −150.',
        cz: 'Pendulum se obvykle naučí dobré vyhoupnutí a udržení za pár set tisíc kroků, takže 200k (★) je rozumný rozpočet. Dejte mu víc, pokud se skóre ještě nešplhá k −150.',
      },
      mountaincarcontinuous: {
        en: 'More steps alone may not solve MountainCarContinuous — with too little exploration the score sits near 0 no matter the budget. Pair a healthy step count with a higher Entropy bonus, or try Neuroevolution.',
        cz: 'Samotné kroky navíc MountainCarContinuous nemusí vyřešit — při příliš malém zkoumání skóre vězí poblíž 0 bez ohledu na rozpočet. Spojte rozumný počet kroků s vyšším bonusem za entropii, nebo zkuste Neuroevoluci.',
      },
      bipedalwalker: {
        en: 'BipedalWalker needs a lot of practice — a smooth gait typically takes a few million steps, hence the large ★ budget. (Training runs on a GPU machine; here Run is disabled until then.)',
        cz: 'BipedalWalker potřebuje hodně cviku — plynulá chůze obvykle zabere pár milionů kroků, odtud velký ★ rozpočet. (Trénink běží na stroji s GPU; tady je Spustit do té doby zakázané.)',
      },
      bipedalwalkerhardcore: {
        en: 'The obstacle course is notoriously hard, so it needs an even larger budget (the ★ 10M) — and may still not fully solve. Training is GPU-only.',
        cz: 'Překážková dráha je pověstně těžká, takže potřebuje ještě větší rozpočet (★ 10M) — a i tak nemusí plně vyřešit. Trénink je jen na GPU.',
      },
      frozenlake: {
        en: 'Slippery and sparse, so give it a generous budget (the ★ 200k). A success rate stuck near 0 means it has not found a route yet — raise the Entropy bonus rather than just adding steps.',
        cz: 'Kluzké a řídké, takže mu dejte štědrý rozpočet (★ 200k). Úspěšnost zaseknutá poblíž 0 znamená, že ještě nenašlo cestu — zvyšte raději bonus za entropii než jen přidávat kroky.',
      },
      frozenlake_noslip: {
        en: 'Deterministic and tiny, so it solves fast — the ★ 50k is plenty.',
        cz: 'Deterministické a drobné, takže se vyřeší rychle — ★ 50k bohatě stačí.',
      },
      frozenlake8x8: {
        en: 'The hardest FrozenLake — pick the largest budget (the ★ 400k) and expect slow early progress while it explores the big lake.',
        cz: 'Nejtěžší FrozenLake — zvolte největší rozpočet (★ 400k) a počítejte s pomalým začátkem, než prozkoumá velké jezero.',
      },
      taxi: {
        en: 'Taxi needs lots of practice (500 states) — the ★ 500k is a starting point. Give it more if the score is not yet climbing past 0 toward +8.',
        cz: 'Taxi potřebuje hodně cviku (500 stavů) — ★ 500k je výchozí bod. Dejte mu víc, pokud se skóre ještě nešplhá přes 0 k +8.',
      },
      cliffwalking: {
        en: 'Usually learns a good path within the ★ 200k; the dense −1 reward makes progress steady, so you will see it climb out of the deep negatives fairly quickly.',
        cz: 'Dobrou cestu se obvykle naučí během ★ 200k; hustá odměna −1 dělá pokrok plynulým, takže ho uvidíte vyšplhat z hlubokého záporu poměrně rychle.',
      },
      minigrid_empty: {
        en: 'Solves within the ★ 100k, often in far less (~25k) — the success curve jumps to ~0.95 quickly once it finds the goal.',
        cz: 'Vyřeší se během ★ 100k, často za mnohem méně (~25k) — křivka úspěšnosti vyskočí k ~0,95 rychle, jakmile cíl najde.',
      },
      minigrid_fourrooms: {
        en: 'Needs a generous budget (★ 500k) for the exploration. A flat curve near 0 early on is normal — it pays nothing until it first reaches the randomly-placed goal.',
        cz: 'Potřebuje štědrý rozpočet (★ 500k) na zkoumání. Rovná křivka poblíž 0 na začátku je normální — nic neplatí, dokud poprvé nedosáhne náhodně umístěného cíle.',
      },
      minigrid_doorkey: {
        en: 'The ★ 300k is a starting point; the key → door → goal sequence takes practice, so give it more if the success curve has not begun to climb.',
        cz: '★ 300k je výchozí bod; posloupnost klíč → dveře → cíl chce cvik, takže přidejte víc, pokud křivka úspěšnosti ještě nezačala stoupat.',
      },
      minigrid_keycorridor: {
        en: 'The hardest of these — choose the largest budget (★ 500k) and expect a long flat start. It may still not fully solve at this budget; more steps and Entropy help.',
        cz: 'Nejtěžší z nich — zvolte největší rozpočet (★ 500k) a počítejte s dlouhým rovným začátkem. I tak nemusí při tomto rozpočtu plně vyřešit; pomohou další kroky a entropie.',
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
      pendulum: {
        en: '50 networks find a swing-up within a reasonable number of generations; a larger population helps a little, at a proportional cost in time.',
        cz: '50 sítí najde vyhoupnutí Pendula za rozumný počet generací; větší populace pomůže o něco víc, za úměrnou cenu v čase.',
      },
      mountaincarcontinuous: {
        en: 'A bigger population is a real advantage here: with more networks tried per generation, one is far more likely to reach the flag and seed the whole herd. Lean toward the upper end (e.g. 100).',
        cz: 'Větší populace je tu skutečná výhoda: s víc sítěmi za generaci je mnohem pravděpodobnější, že některá dojede k vlajce a „nasází“ celé stádo. Posuňte se k horní hranici (např. 100).',
      },
      frozenlake: {
        en: 'A bigger population tries more routes per generation, raising the odds one finds the goal on the slippery ice — lean higher (e.g. 100).',
        cz: 'Větší populace zkouší za generaci víc cest a zvyšuje šanci, že některá najde cíl na kluzkém ledu — posuňte se výš (např. 100).',
      },
      frozenlake_noslip: {
        en: 'Even 30–50 networks find the short safe path quickly on this easy deterministic map.',
        cz: 'I 30–50 sítí najde na této snadné deterministické mapě krátkou bezpečnou cestu rychle.',
      },
      frozenlake8x8: {
        en: 'Lean to the upper end (100+): more networks per generation improves the odds of crossing the bigger lake at all.',
        cz: 'Posuňte se k horní hranici (100+): víc sítí za generaci zvyšuje šanci, že větší jezero vůbec přejde.',
      },
      taxi: {
        en: 'Taxi\'s 500 states are hard for the tiny genome, so a larger population helps a little — but PPO is the better tool here.',
        cz: '500 stavů Taxi je pro drobný genom těžkých, takže větší populace trochu pomůže — lepším nástrojem je tu ale PPO.',
      },
      cliffwalking: {
        en: '50 networks find a path within a reasonable number of generations; a larger population helps a little, at a proportional cost in time.',
        cz: '50 sítí najde cestu za rozumný počet generací; větší populace pomůže o něco víc, za úměrnou cenu v čase.',
      },
      minigrid_empty: {
        en: 'Even ~50 networks can find the small room\'s goal, though the ~2835-input genome is heavy. Lean higher if it stalls.',
        cz: 'I ~50 sítí najde cíl malé místnosti, i když genom s ~2835 vstupy je těžký. Posuňte se výš, pokud učení vázne.',
      },
      minigrid_fourrooms: {
        en: 'Lean to the upper end (100+): more networks per generation improve the odds one finds the goal. The big observation keeps evolution weak here regardless, so PPO is the better tool.',
        cz: 'Posuňte se k horní hranici (100+): víc sítí za generaci zvyšuje šanci, že některá najde cíl. Velký obs tu i tak drží evoluci slabou, takže lepším nástrojem je PPO.',
      },
      minigrid_doorkey: {
        en: 'A larger population helps a little, but the multi-step key → door task is hard for the tiny genome on this big observation — PPO is the better tool here.',
        cz: 'Větší populace trochu pomůže, ale vícekroková úloha klíč → dveře je pro drobný genom na tomto velkém obs těžká — lepším nástrojem je tu PPO.',
      },
      minigrid_keycorridor: {
        en: 'Even a big population rarely cracks this hierarchical task with a small genome on a 2835-dim observation. Included for comparison, but PPO is the practical choice.',
        cz: 'I velká populace zřídka rozlouskne tuto hierarchickou úlohu s malým genomem na obs o 2835 rozměrech. Je tu pro srovnání, ale praktickou volbou je PPO.',
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
      pendulum: {
        en: 'The top ~10 keeps enough diversity for Pendulum to refine the swing-up across generations.',
        cz: 'Horních ~10 udrží pro Pendulum dost rozmanitosti, aby napříč generacemi vyladil vyhoupnutí.',
      },
      mountaincarcontinuous: {
        en: 'Keep enough parents (~10) so one lucky flag-reacher does not dominate before the strategy is reliable. Too few and the herd narrows onto a single fluke.',
        cz: 'Ponechte dost rodičů (~10), aby jeden šťastlivec u vlajky neovládl populaci dřív, než je strategie spolehlivá. Příliš málo a stádo se zúží na jednu náhodu.',
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
      pendulum: {
        en: '0.1 balances exploration and stability for Pendulum. Too high and good swing-ups get scrambled every generation.',
        cz: '0.1 vyvažuje u Pendula zkoumání a stabilitu. Příliš vysoká a dobrá vyhoupnutí se každou generaci rozházejí.',
      },
      mountaincarcontinuous: {
        en: 'A slightly higher rate can help the population explore enough to find the flag at all; once some networks reach it, 0.1 refines them well.',
        cz: 'Mírně vyšší míra může populaci pomoci dostatečně zkoumat, aby vlajku vůbec našla; jakmile k ní některé sítě dojedou, 0,1 je dobře doladí.',
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
      pendulum: {
        en: '0.5 is a safe default for Pendulum; mixing parents spreads a good swing-up rhythm through the herd.',
        cz: '0.5 je pro Pendulum bezpečná výchozí hodnota; mísení rodičů šíří dobrý rytmus vyhoupnutí napříč stádem.',
      },
      mountaincarcontinuous: {
        en: 'Mixing parents spreads a rare flag-reaching trick through the population, so 0.5 is a safe default — most useful once at least one network has succeeded.',
        cz: 'Mísení rodičů šíří vzácný trik, jak dojet k vlajce, napříč populací, takže 0,5 je bezpečná výchozí hodnota — nejužitečnější, jakmile uspěje aspoň jedna síť.',
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
      pendulum: {
        en: 'Pendulum improves steadily over a few dozen generations; push higher to keep refining the swing-up and the balancing hold.',
        cz: 'Pendulum se plynule zlepšuje během pár desítek generací; vyšší hodnotou ho necháte dál ladit vyhoupnutí a udržení rovnováhy.',
      },
      mountaincarcontinuous: {
        en: 'It can take many generations before a network first reaches the flag — then progress accelerates. Give it plenty (e.g. 50+) and a larger population to improve the odds.',
        cz: 'Než nějaká síť poprvé dojede k vlajce, může to trvat řadu generací — pak se pokrok zrychlí. Dejte jí dost (např. 50+) a větší populaci, ať zvýšíte šance.',
      },
      frozenlake: {
        en: 'Give it plenty (50+) and a larger population — it can take many generations before a network first reaches the goal on the slippery ice, after which the herd refines quickly.',
        cz: 'Dejte jí dost (50+) a větší populaci — než nějaká síť poprvé dojde do cíle na kluzkém ledu, může to trvat řadu generací; poté se stádo rychle doladí.',
      },
      frozenlake_noslip: {
        en: 'Often solved within a few dozen generations on this easy deterministic map.',
        cz: 'Na této snadné deterministické mapě bývá vyřešeno během pár desítek generací.',
      },
      frozenlake8x8: {
        en: 'Push it high (100+) — the bigger lake needs many rounds to discover and refine a crossing.',
        cz: 'Posuňte hodně vysoko (100+) — větší jezero potřebuje mnoho kol, než objeví a doladí přechod.',
      },
      taxi: {
        en: 'Many generations are needed and the tiny genome may still not master Taxi\'s 500 states — PPO is the faster route here.',
        cz: 'Je potřeba mnoho generací a drobný genom 500 stavů Taxi stejně nemusí zvládnout — rychlejší cestou je tu PPO.',
      },
      cliffwalking: {
        en: 'Improves steadily over a few dozen generations as the population learns to avoid the cliff; push higher to keep refining toward the optimal path.',
        cz: 'Plynule se zlepšuje během pár desítek generací, jak se populace učí vyhýbat útesu; vyšší hodnotou ji necháte dál ladit k optimální cestě.',
      },
      minigrid_empty: {
        en: 'Improves over a few dozen generations on this easy room; push higher to keep refining toward a short route to the goal.',
        cz: 'Zlepšuje se během pár desítek generací na této snadné místnosti; vyšší hodnotou ji necháte dál ladit ke krátké cestě k cíli.',
      },
      minigrid_fourrooms: {
        en: 'Needs many generations and may still explore poorly — the ~2835-dim observation limits the tiny genome here.',
        cz: 'Potřebuje mnoho generací a stejně může zkoumat špatně — obs o ~2835 rozměrech tu drobný genom omezuje.',
      },
      minigrid_doorkey: {
        en: 'Many generations are needed, and the tiny network often plateaus before mastering the key → door sequence. PPO is the faster route.',
        cz: 'Je potřeba mnoho generací a drobná síť často uvázne, než zvládne posloupnost klíč → dveře. Rychlejší cestou je PPO.',
      },
      minigrid_keycorridor: {
        en: 'Even many generations rarely solve this hierarchical task with evolution — expect a low plateau. PPO is the practical choice.',
        cz: 'I mnoho generací tuto hierarchickou úlohu evolucí jen zřídka vyřeší — čekejte nízké uváznutí. Praktickou volbou je PPO.',
      },
    },
  },

  // ── Tabular Q-learning settings (G2b) ─────────────────────────────────────
  // The value-based 3rd algorithm: a [states × actions] table refined by the Bellman
  // update, available on the discrete grid-world games. These pair with the live Q-table
  // heatmap in the bottom panel.

  q_learning_rate: {
    general: {
      en: '**Learning rate (α)** — how much each new experience updates the table.\nQ-learning nudges a cell toward what it just observed; α is the size of that nudge. Higher → learns fast but jumpily (noisy estimates). Lower → smooth but slow. This is a *table* step, so it is far larger than PPO\'s neural-network learning rate.',
      cz: '**Rychlost učení (α)** — jak moc každá nová zkušenost upraví tabulku.\nQ-učení posune buňku směrem k tomu, co právě pozorovalo; α je velikost tohoto posunu. Vyšší → učí se rychle, ale trhaně (zašuměné odhady). Nižší → plynule, ale pomalu. Jde o krok *tabulky*, takže je mnohem větší než rychlost učení neuronové sítě u PPO.',
    },
    recommended: {
      en: '0.1 — a steady default. On the deterministic (no-slip) maps you can go higher (0.2–0.5) to learn faster; on slippery FrozenLake keep it moderate so noise does not destabilise the estimates.',
      cz: '0.1 — stabilní výchozí hodnota. Na deterministických (neklouzavých) mapách můžete jít výš (0,2–0,5) pro rychlejší učení; na kluzkém FrozenLake ji držte umírněnou, aby šum nerozhodil odhady.',
    },
    range: '0.01 – 1.0',
  },

  epsilon_start: {
    general: {
      en: '**Starting exploration (ε).** Q-learning is ε-greedy: with probability ε it tries a *random* action instead of the current best one. ε starts high so the agent explores widely before it knows anything.\n1.0 = "act completely at random at first".',
      cz: '**Počáteční zkoumání (ε).** Q-učení je ε-hladové: s pravděpodobností ε zkusí *náhodnou* akci místo aktuálně nejlepší. ε začíná vysoko, aby agent zpočátku hodně zkoumal, než cokoli ví.\n1.0 = „na začátku jednej zcela náhodně“.',
    },
    recommended: {
      en: '1.0 — start fully exploratory. The table is empty at the start, so there is nothing to exploit yet; explore everything first.',
      cz: '1.0 — začněte plně průzkumně. Tabulka je na začátku prázdná, takže není co využívat; nejdřív vše prozkoumejte.',
    },
    range: '0.1 – 1.0',
  },

  epsilon_end: {
    general: {
      en: '**Final exploration (ε).** As the table fills in, the agent should rely more on what it has learned and explore less. ε anneals down to this floor and then holds.\nA small non-zero value keeps a little exploration so the agent can still discover a better route.',
      cz: '**Konečné zkoumání (ε).** Jak se tabulka plní, agent by se měl víc spoléhat na to, co se naučil, a méně zkoumat. ε klesá k této spodní hranici a tam zůstane.\nMalá nenulová hodnota zachová trochu zkoumání, aby agent ještě mohl objevit lepší cestu.',
    },
    recommended: {
      en: '0.05 — mostly exploit the learned table, but keep a 5% trickle of exploration.',
      cz: '0.05 — převážně využívej naučenou tabulku, ale ponech 5% pramínek zkoumání.',
    },
    range: '0.0 – 0.5',
  },

  epsilon_decay: {
    general: {
      en: '**Exploration schedule** — over what fraction of the episode budget ε falls from its start value to its end value (then holds).\n0.5 = "spend the first half of training annealing exploration, then stay greedy". Smaller = settle down sooner; larger = keep exploring for longer.',
      cz: '**Plán zkoumání** — přes jakou část rozpočtu epizod klesne ε z počáteční hodnoty na konečnou (a tam zůstane).\n0.5 = „první polovinu tréninku utlumuj zkoumání, pak zůstaň hladový“. Menší = usadí se dřív; větší = zkoumá déle.',
    },
    recommended: {
      en: '0.5 — anneal over the first half of training. Raise it for hard, slippery maps that need to keep exploring; lower it for easy deterministic ones that can settle quickly.',
      cz: '0.5 — utlumuj přes první polovinu tréninku. Zvyšte ji u těžkých, kluzkých map, které musí dál zkoumat; snižte u snadných deterministických, které se mohou rychle usadit.',
    },
    range: '0.1 – 1.0',
  },

  episodes: {
    general: {
      en: '**Training budget** — how many full games (episodes) Q-learning plays to fill in its table.\nMore episodes mean a more complete, better-refined table, up to the point the game is mastered. This is Q-learning\'s equivalent of PPO\'s Total Steps.',
      cz: '**Tréninkový rozpočet** — kolik celých her (epizod) Q-učení odehraje, aby naplnilo tabulku.\nVíce epizod znamená úplnější, lépe vyladěnou tabulku, dokud hru nezvládne. Je to obdoba PPO „Celkem kroků“ pro Q-učení.',
    },
    recommended: {
      en: 'The ★ default is tuned per game (a small deterministic maze needs a few thousand; Taxi\'s 500 states want far more). Watch the Filled % and the reward curve — if they are still climbing, add episodes.',
      cz: 'Výchozí ★ je laděná pro každou hru (malé deterministické bludiště potřebuje pár tisíc; 500 stavů Taxi mnohem víc). Sledujte Zaplněno % a křivku odměny — pokud stále stoupají, přidejte epizody.',
    },
    range: '500 – 50000  (depends on the game)',
  },

  qtable: {
    general: {
      en: 'This panel is the **Q-table** itself — Q-learning\'s entire brain, shown live.\nEach **row is a state** (a grid cell / a Taxi situation) and each **column is an action** (the arrows; Taxi adds Pickup/Drop-off). A cell\'s colour is its learned value: **green = good**, **red = bad**, **blank = not learned yet** — so you watch the table light up as training explores. The **outlined cell** in each row is the action the agent would currently take there (its policy).\n**Episode** = games played / budget. **ε** = current exploration rate (falls over training). **Filled** = fraction of the table touched so far. **Score** = recent mean return.',
      cz: 'Tento panel je samotná **Q-tabulka** — celý „mozek“ Q-učení, živě.\nKaždý **řádek je stav** (buňka mřížky / situace Taxíku) a každý **sloupec je akce** (šipky; Taxi přidává Naber/Vysaď). Barva buňky je její naučená hodnota: **zelená = dobrá**, **červená = špatná**, **prázdná = zatím nenaučená** — takže sledujete, jak se tabulka rozsvěcí, jak trénink zkoumá. **Orámovaná buňka** v každém řádku je akce, kterou by tam agent právě zvolil (jeho strategie).\n**Epizoda** = odehrané hry / rozpočet. **ε** = aktuální míra zkoumání (klesá během tréninku). **Zaplněno** = podíl dosud dotčené tabulky. **Skóre** = nedávná průměrná odměna.',
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
      pendulum: {
        en: "Pendulum's reward is a per-step penalty (how far from upright, how fast it spins, how hard you push), so the score is always negative — the curve climbs from about −1600 toward 0, and a good swing-up-and-hold reaches around −150.",
        cz: 'Odměna u Pendula je penalizace za krok (jak daleko od svislé polohy, jak rychle se točí, jak silně tlačíte), takže skóre je vždy záporné — křivka stoupá zhruba z −1600 k 0 a dobré vyhoupnutí a udržení dosáhne kolem −150.',
      },
      mountaincarcontinuous: {
        en: 'MountainCarContinuous pays +100 for reaching the flag minus a small force cost, so the curve sits near 0 until the agent first reaches the flag, then jumps toward +90. A flat line near 0 means it has not found the flag yet — the exploration problem, not a bug.',
        cz: 'MountainCarContinuous vyplácí +100 za dosažení vlajky minus malou cenu za sílu, takže křivka sedí poblíž 0, dokud agent poprvé nedojede k vlajce, pak vyskočí k +90. Rovná čára poblíž 0 znamená, že vlajku ještě nenašel — to je ten problém se zkoumáním, ne chyba.',
      },
      bipedalwalker: {
        en: 'BipedalWalker pays a little for each bit of forward progress, costs a little for using the motors, and −100 for a fall, so the curve climbs from about −100 toward +300 as the agent learns to walk; +300 means a clean walk to the far end ("solved").',
        cz: 'BipedalWalker platí trochu za každý kousek postupu vpřed, něco stojí použití motorů a −100 za pád, takže křivka stoupá zhruba z −100 k +300, jak se agent učí chodit; +300 znamená čistou chůzi až na konec („vyřešeno“).',
      },
      bipedalwalkerhardcore: {
        en: 'Same reward shape, but progress is much slower and the curve stays low far longer while the agent learns to clear the obstacles; +300 (finishing) is hard to reach.',
        cz: 'Stejný tvar odměny, ale postup je mnohem pomalejší a křivka zůstává nízko mnohem déle, než se agent naučí překonávat překážky; +300 (dojití) je těžké dosáhnout.',
      },
      frozenlake: {
        en: 'FrozenLake pays 1 only for reaching the goal, else 0, so each episode scores 0 or 1 and the training curve is the success *rate*. "Solved" is 0.70 — the curve climbs from ~0 toward 0.7+, and cannot reach 1.0 because the ice sometimes slips you into a hole.',
        cz: 'FrozenLake platí 1 jen za dosažení cíle, jinak 0, takže každá epizoda dá 0 nebo 1 a tréninková křivka je *míra* úspěšnosti. „Vyřešeno“ je 0,70 — křivka stoupá z ~0 k 0,7+ a nemůže dosáhnout 1,0, protože led vás občas smekne do díry.',
      },
      frozenlake_noslip: {
        en: 'Reward is 1 for the goal, else 0. With no slipping a good policy reaches the goal every time, so the success-rate curve can climb all the way to 1.0; "solved" is 0.70.',
        cz: 'Odměna je 1 za cíl, jinak 0. Bez klouzání dojde dobrá strategie do cíle pokaždé, takže křivka úspěšnosti může vyšplhat až na 1,0; „vyřešeno“ je 0,70.',
      },
      frozenlake8x8: {
        en: 'Same 0/1 reward → the curve is the success rate, climbing from ~0 toward 0.7+ on the bigger, slippier lake. A flat line at 0 means it has not found the goal yet — raise exploration.',
        cz: 'Stejná odměna 0/1 → křivka je míra úspěšnosti, stoupající z ~0 k 0,7+ na větším, kluzčím jezeře. Rovná čára na 0 znamená, že cíl ještě nenašlo — zvyšte zkoumání.',
      },
      taxi: {
        en: 'Taxi gives −1 per step, +20 for a correct drop-off and −10 for an illegal pickup/drop-off, so early scores are very negative (−200 to −800) and climb toward +8 ("solved") as the agent stops wasting moves and illegal actions.',
        cz: 'Taxi dává −1 za krok, +20 za správné vysazení a −10 za nelegální nabrání/vysazení, takže rané skóre je hodně záporné (−200 až −800) a stoupá k +8 („vyřešeno“), jak agent přestává plýtvat kroky a dělat nelegální akce.',
      },
      cliffwalking: {
        en: 'CliffWalking gives −1 per step and −100 per cliff fall, so a flailing agent scores in the thousands negative; the curve climbs toward about −13 (the optimal path) as it learns to avoid the cliff.',
        cz: 'CliffWalking dává −1 za krok a −100 za pád z útesu, takže zmatený agent skóruje v tisících záporných; křivka stoupá zhruba k −13 (optimální cesta), jak se učí útesu vyhýbat.',
      },
      minigrid_empty: {
        en: 'MiniGrid pays nothing until the goal, then 1 − 0.9·(steps/max) on success (≈0.9–0.97), so the curve sits near 0 until it first reaches the goal, then climbs to ~0.95 ("solved").',
        cz: 'MiniGrid neplatí nic až do cíle, pak při úspěchu 1 − 0,9·(kroky/max) (≈0,9–0,97), takže křivka sedí poblíž 0, dokud cíl poprvé nedosáhne, pak stoupá k ~0,95 („vyřešeno“).',
      },
      minigrid_fourrooms: {
        en: 'Same sparse reward — the curve is flat near 0 until exploration finds the goal, then rises. A long flat line means it has not found the goal yet; raise exploration.',
        cz: 'Stejná řídká odměna — křivka je rovná poblíž 0, dokud zkoumání nenajde cíl, pak stoupá. Dlouhá rovná čára znamená, že cíl ještě nenašlo; zvyšte zkoumání.',
      },
      minigrid_doorkey: {
        en: 'Same shape — it stays near 0 until the full key → door → goal is completed, then climbs. A flat line means the sequence is not solved yet.',
        cz: 'Stejný tvar — drží se poblíž 0, dokud se nedokončí celé klíč → dveře → cíl, pak stoupá. Rovná čára znamená, že posloupnost ještě není vyřešená.',
      },
      minigrid_keycorridor: {
        en: 'Same sparse reward; expect a long flat 0 (the ball is hard to reach) before any climb — that is the sparse-reward profile, not a bug.',
        cz: 'Stejná řídká odměna; čekejte dlouhou rovnou 0 (na míček je těžké dosáhnout), než vůbec začne stoupat — to je profil řídké odměny, ne chyba.',
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
      pendulum: {
        en: 'As always, the loss just wobbles — judge Pendulum progress from the reward curve climbing toward 0, not from this one.',
        cz: 'Jako vždy ztráta jen kolísá — pokrok u Pendula posuzujte podle křivky odměny stoupající k 0, ne podle ní.',
      },
      mountaincarcontinuous: {
        en: 'The loss just wobbles here too; read progress from the reward curve (has it jumped toward +90?), not from the loss.',
        cz: 'Ztráta i tady jen kolísá; pokrok čtěte z křivky odměny (vyskočila k +90?), ne ze ztráty.',
      },
      bipedalwalker: {
        en: 'As always the loss just wobbles — judge BipedalWalker progress from the reward curve climbing toward +300, not from this one. Wild spikes can hint at instability (lower the learning rate).',
        cz: 'Jako vždy ztráta jen kolísá — pokrok u BipedalWalkeru posuzujte podle křivky odměny stoupající k +300, ne podle ní. Divoké výkyvy mohou naznačovat nestabilitu (snižte rychlost učení).',
      },
      bipedalwalkerhardcore: {
        en: 'The loss just wobbles here too; read progress from the reward curve, not the loss.',
        cz: 'Ztráta i tady jen kolísá; pokrok čtěte z křivky odměny, ne ze ztráty.',
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
      pendulum: {
        en: 'Same idea as reward, but per generation. For Pendulum fitness starts deeply negative and rises toward −150 as the population learns to swing up and hold.',
        cz: 'Stejný princip jako odměna, ale za generaci. U Pendula začíná fitness hluboko v záporu a stoupá k −150, jak se populace učí vyhoupnout a udržet.',
      },
      mountaincarcontinuous: {
        en: "Same idea as reward, but per generation. Fitness sits near 0 until some network first reaches the flag, then jumps up — neuroevolution's population search is well suited to making that discovery.",
        cz: 'Stejný princip jako odměna, ale za generaci. Fitness sedí poblíž 0, dokud nějaká síť poprvé nedojede k vlajce, pak vyskočí — populační hledání neuroevoluce se k takovému objevu dobře hodí.',
      },
      frozenlake: {
        en: 'Same as reward but per generation — the population\'s success rate, climbing from ~0 toward 0.7 (it cannot reach 1.0 because the ice slips).',
        cz: 'Stejné jako odměna, ale za generaci — míra úspěšnosti populace, stoupající z ~0 k 0,7 (nemůže dosáhnout 1,0, protože led klouže).',
      },
      frozenlake_noslip: {
        en: 'Per-generation success rate; on this easy deterministic map it can climb all the way to 1.0.',
        cz: 'Míra úspěšnosti za generaci; na této snadné deterministické mapě může vyšplhat až na 1,0.',
      },
      frozenlake8x8: {
        en: 'Per-generation success rate on the big lake; it rises slowly and, with the tiny genome, often plateaus below the 0.7 "solved" line.',
        cz: 'Míra úspěšnosti za generaci na velkém jezeře; stoupá pomalu a s drobným genomem často uvázne pod hranicí 0,7 („vyřešeno“).',
      },
      taxi: {
        en: 'Per generation; starts very negative and rises toward +8 — though the tiny evolution network struggles with Taxi\'s 500 states, so it often plateaus well short.',
        cz: 'Za generaci; začíná hodně záporně a stoupá k +8 — drobná evoluční síť ale s 500 stavy Taxi bojuje, takže často uvázne daleko před cílem.',
      },
      cliffwalking: {
        en: 'Per generation; climbs from deeply negative toward about −13 as the population learns to avoid the cliff and take a short path.',
        cz: 'Za generaci; stoupá z hluboce záporných hodnot zhruba k −13, jak se populace učí vyhnout útesu a jít krátkou cestou.',
      },
      minigrid_empty: {
        en: 'Same as reward but per generation — rises from ~0 toward ~0.95 once a network finds the goal.',
        cz: 'Stejné jako odměna, ale za generaci — stoupá z ~0 k ~0,95, jakmile některá síť najde cíl.',
      },
      minigrid_fourrooms: {
        en: 'Per-generation success reward; it rises slowly and, with the tiny genome on the big observation, often plateaus low.',
        cz: 'Odměna za úspěch za generaci; stoupá pomalu a s drobným genomem na velkém obs často uvázne nízko.',
      },
      minigrid_doorkey: {
        en: 'Per generation; often plateaus below "solved" as the small network struggles with the key → door sub-goal.',
        cz: 'Za generaci; často uvázne pod „vyřešeno“, jak malá síť bojuje s dílčím cílem klíč → dveře.',
      },
      minigrid_keycorridor: {
        en: 'Per generation; usually stays near 0 — neuroevolution rarely cracks this hierarchical task.',
        cz: 'Za generaci; obvykle zůstává poblíž 0 — neuroevoluce tuto hierarchickou úlohu jen zřídka rozlouskne.',
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
