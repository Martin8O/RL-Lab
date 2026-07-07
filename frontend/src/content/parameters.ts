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

// MuJoCo (G5a) — the six continuous-control robots share almost all of their parameter guidance
// (continuous box action, MlpPolicy, GPU-gated training), so the env-agnostic notes are defined once
// here and referenced from each robot's perEnv key below. Only `reward` (whose numbers differ per
// robot) is written per-env. Mirrors the shared-constant pattern in playGuides.ts (BIPEDAL_* etc.).
const MUJOCO_ALGO: Bilingual = {
  en: 'Three algorithms here: **PPO**, **SAC** and **TD3**. These MuJoCo robots are continuous-control '
    + 'tasks — the agent outputs smooth joint torques, not button presses. **PPO** (on-policy) learns a '
    + 'gait but needs many millions of steps. **SAC** and **TD3** are both **off-policy** (they keep a '
    + 'replay buffer of past experience and reuse it), so they are far more sample-efficient and shine on '
    + 'MuJoCo — SAC explores via entropy, TD3 via a deterministic policy plus injected action noise. A '
    + 'great head-to-head: train two of them on the same robot and compare. Neuroevolution is turned off '
    + 'as data (population search is impractical on hard multi-joint control). Training takes a lot of '
    + 'compute, so it is reserved for a GPU machine; you can play it by hand now and watch a trained AI.',
  cz: 'Tady jsou tři algoritmy: **PPO**, **SAC** a **TD3**. Tito roboti MuJoCo jsou úlohy spojitého řízení '
    + '— agent vydává plynulé momenty v kloubech, ne stisky tlačítek. **PPO** (on-policy) se chůzi naučí, '
    + 'ale potřebuje mnoho milionů kroků. **SAC** i **TD3** jsou **off-policy** (drží si paměť minulých '
    + 'zkušeností — replay buffer — a znovu ji využívají), takže jsou mnohem úspornější na data a na MuJoCo '
    + 'vynikají — SAC zkoumá pomocí entropie, TD3 pomocí deterministické strategie a přidaného šumu do '
    + 'akcí. Skvělý souboj: natrénujte dva z nich na stejném robotovi a porovnejte. Neuroevoluce je '
    + 'vypnutá jako data (populační hledání je u těžkého víceklouobového řízení nepraktické). Trénink je '
    + 'výpočetně náročný, takže je vyhrazen pro stroj s GPU; rukama si to zahrajete už teď a natrénovanou '
    + 'AI můžete sledovat.',
}

const MUJOCO_LR: Bilingual = {
  en: 'The default 3e-4 is the standard starting point for MuJoCo too. Continuous control can be '
    + 'sensitive, so keep it moderate — too large a step destabilises the gait; lean on more steps '
    + 'instead. (Training is GPU-only, so this applies once you train on the desktop.)',
  cz: 'Výchozí 3e-4 je standardní výchozí bod i pro MuJoCo. Spojité řízení může být citlivé, takže ji '
    + 'držte umírněnou — příliš velký krok rozhodí chůzi; spoléhejte raději na víc kroků. (Trénink je '
    + 'jen na GPU, takže se to týká až tréninku na desktopu.)',
}

const MUJOCO_GAMMA: Bilingual = {
  en: 'Keep γ high (0.99): locomotion is a long sequence of coordinated steps, so the agent must value '
    + 'progress and staying upright many steps ahead, not just the next torque.',
  cz: 'Nechte γ vysoké (0.99): pohyb je dlouhá řada koordinovaných kroků, takže agent musí cenit postup '
    + 'a udržení vzpřímené polohy mnoho kroků dopředu, ne jen další moment.',
}

const MUJOCO_CLIP: Bilingual = {
  en: 'Leave it at 0.2 — continuous control can be unstable, and the clip is what keeps one bad batch '
    + 'from wrecking a working gait; tune budget and learning rate first.',
  cz: 'Nechte ji na 0.2 — spojité řízení může být nestabilní a ořezávání brání jedné špatné dávce '
    + 'zničit funkční chůzi; laďte nejdřív rozpočet a rychlost učení.',
}

const MUJOCO_ENT: Bilingual = {
  en: "PPO's continuous action head already explores with its own noise, so 0 can work; a small bonus "
    + '(e.g. 0.01) can help it discover forward motion instead of standing still. Keep it modest — too '
    + 'much randomness just makes the robot flail.',
  cz: 'Spojitá akční hlava PPO už zkoumá vlastním šumem, takže 0 může stačit; drobný bonus (např. 0,01) '
    + 'může pomoci objevit pohyb vpřed místo stání na místě. Držte ho mírný — moc náhodnosti robota jen '
    + 'rozhází.',
}

const MUJOCO_STEPS: Bilingual = {
  en: 'The ★ budget is tuned per robot: the locomotion tasks (Hopper, Walker2d, HalfCheetah, Ant) need '
    + 'a lot of practice — a good gait typically takes a few million steps — while the short Reacher '
    + 'reach needs far fewer. (Training runs on a GPU machine; here Run is disabled until then.)',
  cz: 'Rozpočet ★ je laděný pro každého robota: pohybové úlohy (Hopper, Walker2d, HalfCheetah, Ant) '
    + 'potřebují hodně cviku — dobrá chůze obvykle zabere pár milionů kroků — kdežto krátké dosažení u '
    + 'Reacheru mnohem méně. (Trénink běží na stroji s GPU; tady je Spustit do té doby zakázané.)',
}

const MUJOCO_LOSS: Bilingual = {
  en: 'As always the loss just wobbles — judge MuJoCo progress from the reward curve climbing toward '
    + 'the solved score, not from this one. Wild spikes can hint at instability (lower the learning rate).',
  cz: 'Jako vždy ztráta jen kolísá — pokrok u MuJoCo posuzujte podle křivky odměny stoupající ke skóre '
    + '„vyřešeno“, ne podle ní. Divoké výkyvy mohou naznačovat nestabilitu (snižte rychlost učení).',
}

export const PARAM_INFO: Record<string, ParamInfo> = {
  // ── Data Lab controls (X6a) — analysis-surface concepts, not training params. Env-agnostic, so no
  // perEnv notes; ParamInfo simply renders the general (+recommended) sections for these ids. ──────
  analysis_mode: {
    general: {
      en: 'How the overlaid runs are put on a **common vertical scale** so they can share one chart.\n'
        + '**Per game (raw reward):** the y-axis is the game\'s actual score. Only meaningful when every '
        + 'selected run is the *same game* — different games have wildly different reward scales.\n'
        + '**Per algorithm (skill %):** each run is rescaled to 0–100% of *its own* game\'s skill meter '
        + '(0% = the idle floor, 100% = solved). This normalises across games, so you can honestly '
        + 'compare, say, PPO on CartPole against PPO on LunarLander on one axis.',
      cz: 'Jak se překryté běhy dostanou na **společnou svislou škálu**, aby se vešly do jednoho grafu.\n'
        + '**Podle hry (surová odměna):** osa y je skutečné skóre hry. Dává smysl jen když jsou všechny '
        + 'vybrané běhy *stejná hra* — různé hry mají úplně jiné škály odměn.\n'
        + '**Podle algoritmu (dovednost %):** každý běh se přeškáluje na 0–100 % dovednostního metru *své* '
        + 'hry (0 % = nečinné dno, 100 % = vyřešeno). Tím se srovnají napříč hrami, takže můžete poctivě '
        + 'porovnat třeba PPO na CartPole proti PPO na LunarLanderu na jedné ose.',
    },
    recommended: {
      en: 'Comparing runs of one game → Per game. Comparing across different games → Per algorithm.',
      cz: 'Porovnáváte běhy jedné hry → Podle hry. Porovnáváte napříč různými hrami → Podle algoritmu.',
    },
  },
  analysis_axis: {
    general: {
      en: 'What the **horizontal axis** measures — two honest but different lenses on "how far did the '
        + 'run get".\n'
        + '**Env steps:** progress in environment steps (how much *experience* the agent consumed). This '
        + 'is the fair, hardware-independent axis — the standard way to compare sample-efficiency.\n'
        + '**Wall-clock:** progress in real elapsed time. Reflects how fast the run actually was on this '
        + 'machine — useful for "which finished sooner", but it is sensitive to CPU/GPU, batching and '
        + 'other load, so it is not a fair sample-efficiency comparison.',
      cz: 'Co měří **vodorovná osa** — dva poctivé, ale odlišné pohledy na to, „kam až se běh dostal“.\n'
        + '**Kroky prostředí:** postup v krocích prostředí (kolik *zkušenosti* agent spotřeboval). To je '
        + 'férová osa nezávislá na hardwaru — standardní způsob, jak porovnat úspornost na data.\n'
        + '**Reálný čas:** postup v uplynulém čase. Ukazuje, jak rychlý běh na tomto stroji skutečně byl — '
        + 'hodí se pro „co doběhlo dřív“, ale závisí na CPU/GPU, dávkování a další zátěži, takže to není '
        + 'férové srovnání úspornosti na data.',
    },
    recommended: {
      en: 'Comparing learning quality → Env steps. Comparing raw speed on this machine → Wall-clock.',
      cz: 'Porovnáváte kvalitu učení → Kroky prostředí. Porovnáváte čistou rychlost na stroji → Reálný čas.',
    },
  },
  analysis_collapse: {
    general: {
      en: 'Merge a set of runs that differ **only by random seed** into a single **mean ± confidence '
        + 'band** instead of one line each.\n'
        + 'A single seed can get lucky or unlucky, so one curve tells you little about whether a setting '
        + 'is really better. Averaging several seeds — the shaded band shows the spread — is the honest '
        + 'way to report an RL result. Available only when the selected runs share one game **and** '
        + 'algorithm and there are at least two of them; the seed chips let you drop an outlier.',
      cz: 'Sloučí sadu běhů, které se liší **jen náhodným seedem**, do jednoho **průměru ± pásma '
        + 'spolehlivosti** místo samostatné čáry pro každý.\n'
        + 'Jeden seed může mít štěstí i smůlu, takže jedna křivka moc neřekne, jestli je nastavení '
        + 'opravdu lepší. Zprůměrování více seedů — stínované pásmo ukazuje rozptyl — je poctivý způsob, '
        + 'jak výsledek RL prezentovat. Dostupné jen když vybrané běhy sdílejí jednu hru **a** algoritmus '
        + 'a jsou aspoň dva; přepínače seedů umožní vyřadit odlehlý běh.',
    },
  },
  analysis_table: {
    general: {
      en: 'The **numbers a paper reports**, one row per selected run — a learning curve shows *how*, this '
        + 'table gives the single scalars you rank on. Click any column header to sort by it.\n'
        + '**Final %:** skill at the end of the run (0% = idle floor, 100% = solved).\n'
        + '**AUC:** area under the normalized-skill curve — the *mean* skill across the whole run, so it '
        + 'rewards learning both **fast and high** in one number. The natural overall ranking key.\n'
        + '**Solved @:** how many environment steps it took to first reach the solved score ("—" = never).\n'
        + '**Peak %:** the best skill the run ever reached.\n'
        + '**Collapse %:** how many skill points it gave back from that peak (0 = it never regressed — a '
        + 'high value flags an unstable run that peaked then fell).\n'
        + '**Steps/s:** training throughput on this machine.\n'
        + 'When seeds are collapsed into a band, the top row shows the mean ± standard deviation across '
        + 'them — the honest summary of a multi-seed experiment.',
      cz: '**Čísla, která uvádí odborný článek**, jeden řádek na vybraný běh — křivka ukazuje *jak*, tato '
        + 'tabulka dává jednotlivá čísla, podle kterých řadíš. Klikni na záhlaví sloupce a seřaď podle něj.\n'
        + '**Konečné %:** dovednost na konci běhu (0 % = nečinné dno, 100 % = vyřešeno).\n'
        + '**AUC:** plocha pod normalizovanou křivkou dovednosti — *průměrná* dovednost za celý běh, takže '
        + 'odměňuje učení **rychlé i vysoké** jedním číslem. Přirozený hlavní klíč pro řazení.\n'
        + '**Vyřešeno @:** kolik kroků prostředí trvalo poprvé dosáhnout skóre „vyřešeno“ („—“ = nikdy).\n'
        + '**Vrchol %:** nejlepší dovednost, jaké běh kdy dosáhl.\n'
        + '**Propad %:** kolik bodů dovednosti od toho vrcholu ztratil (0 = neregresoval — vysoká hodnota '
        + 'značí nestabilní běh, který vyvrcholil a pak spadl).\n'
        + '**Kroky/s:** propustnost tréninku na tomto stroji.\n'
        + 'Když jsou seedy sloučené do pásu, horní řádek ukazuje průměr ± směrodatnou odchylku přes ně — '
        + 'poctivý souhrn experimentu s více seedy.',
    },
  },
  analysis_rliable: {
    general: {
      en: 'A **statistically honest summary** of how the selected algorithms compare, following the *rliable* '
        + 'method (Agarwal et al., 2021). With only a handful of seeds, a plain average is easy to mislead '
        + 'with — these estimators are the robust, publishable alternative. Every run is first rescaled to a '
        + '0–1 score (its final skill % ÷ 100).\n'
        + '**IQM (interquartile mean):** the average of the middle 50% of scores — it ignores one lucky or '
        + 'disastrous seed, so it is the recommended headline metric. **Mean / Median** are shown alongside '
        + 'for context. **Opt. gap (optimality gap):** how far short of a perfect 1.0 the method falls '
        + '(*lower is better*).\n'
        + 'Each estimate carries a **95% confidence interval** (the bar) from a *stratified bootstrap* — '
        + 'the range the true value plausibly lies in. Few seeds → a wide bar, and that width is the point: '
        + 'it tells you not to over-read a small difference.\n'
        + '**Performance profile:** the fraction of runs (y) scoring above each threshold τ (x). A curve '
        + 'entirely above another means that method is better *at every bar* — a much stronger claim than a '
        + 'single average. **Probability of improvement:** P(A > B) across shared games — 0.5 is a coin '
        + 'flip, above 0.5 favours A.',
      cz: '**Statisticky poctivý souhrn** toho, jak si vybrané algoritmy stojí, podle metody *rliable* '
        + '(Agarwal a kol., 2021). S pár seedy se prostým průměrem snadno splete — tyto odhady jsou robustní, '
        + 'publikovatelná alternativa. Každý běh se nejdřív přeškáluje na skóre 0–1 (jeho konečná dovednost % '
        + '÷ 100).\n'
        + '**IQM (mezikvartilový průměr):** průměr prostředních 50 % skóre — ignoruje jeden šťastný či '
        + 'katastrofální seed, takže je to doporučená hlavní metrika. **Průměr / Medián** jsou vedle pro '
        + 'kontext. **Opt. mezera (optimality gap):** o kolik metoda zaostává za dokonalou 1,0 (*méně je '
        + 'lépe*).\n'
        + 'Každý odhad nese **95% interval spolehlivosti** (pruh) ze *stratifikovaného bootstrapu* — rozsah, '
        + 'v němž skutečná hodnota věrohodně leží. Málo seedů → široký pruh, a ta šířka je smyslem: říká ti, '
        + 'ať nepřeceňuješ malý rozdíl.\n'
        + '**Výkonnostní profil:** podíl běhů (y) se skóre nad každým prahem τ (x). Křivka celá nad druhou '
        + 'znamená, že ta metoda je lepší *na každé úrovni* — mnohem silnější tvrzení než jeden průměr. '
        + '**Pravděpodobnost zlepšení:** P(A > B) přes sdílené hry — 0,5 je náhoda, nad 0,5 svědčí pro A.',
    },
  },
  analysis_profile: {
    general: {
      en: 'The **performance profile** shows a whole *distribution* of results, not a single number — the '
        + 'honest way to compare methods when a plain average can hide a lot.\n'
        + '**How to read it:** the horizontal axis is a score threshold **τ** (from 0 = worst to 1 = solved); '
        + 'the vertical axis is the **fraction of runs that scored above τ**. So a point at (0.6, 0.8) means '
        + '"80% of this method\'s runs scored better than 0.6". Each algorithm is one line.\n'
        + '**Why it beats a bar chart:** it uses *every* run at *every* threshold, so one lucky or unlucky '
        + 'seed can\'t swing it, and nobody gets to cherry-pick the threshold that flatters their method.\n'
        + '**The key pattern — dominance:** if one curve sits **entirely above** another, that method has more '
        + 'runs above the bar *at every score* — a far stronger claim than "higher mean". If the curves '
        + '**cross**, neither method is uniformly better: one wins on easy thresholds, the other on hard ones.',
      cz: '**Výkonnostní profil** ukazuje celé *rozdělení* výsledků, ne jedno číslo — poctivý způsob, jak '
        + 'porovnat metody, když prostý průměr může hodně skrýt.\n'
        + '**Jak ho číst:** vodorovná osa je práh skóre **τ** (od 0 = nejhorší po 1 = vyřešeno); svislá osa je '
        + '**podíl běhů, které dosáhly skóre nad τ**. Bod v (0,6, 0,8) tedy znamená „80 % běhů této metody '
        + 'mělo skóre lepší než 0,6“. Každý algoritmus je jedna čára.\n'
        + '**Proč je lepší než sloupcový graf:** používá *každý* běh na *každém* prahu, takže jeden šťastný '
        + 'nebo nešťastný seed s ním nezamává a nikdo si nemůže vybrat práh, který jeho metodě lichotí.\n'
        + '**Klíčový vzor — dominance:** když jedna křivka leží **celá nad** druhou, ta metoda má víc běhů nad '
        + 'laťkou *na každém skóre* — mnohem silnější tvrzení než „vyšší průměr“. Když se křivky **kříží**, '
        + 'ani jedna metoda není lepší všude: jedna vítězí na snadných prazích, druhá na těžkých.',
    },
    recommended: {
      en: 'Look for a curve that never dips below the others — that method dominates. Crossing curves mean '
        + '"it depends"; report both.',
      cz: 'Hledejte křivku, která nikdy neklesne pod ostatní — ta metoda dominuje. Křížící se křivky znamenají '
        + '„záleží“; uveďte obě.',
    },
  },
  analysis_poi: {
    general: {
      en: 'The **probability of improvement** answers one blunt question: *if I pick a run of A and a run of B '
        + 'at random, how often does A win?* It is P(A > B) estimated across the games the two share (a '
        + 'Mann–Whitney comparison).\n'
        + '**The scale:** **0.5 is a coin flip** — no difference. Above 0.5 favours the first algorithm, below '
        + '0.5 the second. 0.75 means A beats B three times out of four.\n'
        + '**Read the interval, not just the number:** the value carries a 95% bootstrap confidence interval. '
        + 'If that interval **straddles 0.5**, the edge isn\'t statistically clear yet — you likely need more '
        + 'seeds before claiming a winner.\n'
        + '**A caveat:** this measures *how often* A wins, not *by how much*. A can win 60% of the time by a '
        + 'hair, so read it together with the IQM (which shows the size of the gap). Needs at least two '
        + 'algorithms that were run on a shared game.',
      cz: '**Pravděpodobnost zlepšení** odpovídá na jednu přímou otázku: *když náhodně vyberu běh A a běh B, '
        + 'jak často vyhraje A?* Je to P(A > B) odhadnutá přes hry, které oba sdílejí (porovnání '
        + 'Mann–Whitney).\n'
        + '**Škála:** **0,5 je hod mincí** — žádný rozdíl. Nad 0,5 svědčí pro první algoritmus, pod 0,5 pro '
        + 'druhý. 0,75 znamená, že A porazí B ve třech ze čtyř případů.\n'
        + '**Čtěte interval, ne jen číslo:** hodnota nese 95% bootstrapový interval spolehlivosti. Pokud '
        + 'interval **přesahuje přes 0,5**, náskok zatím není statisticky jasný — nejspíš potřebujete víc '
        + 'seedů, než vyhlásíte vítěze.\n'
        + '**Upozornění:** měří to, *jak často* A vyhrává, ne *o kolik*. A může vyhrávat v 60 % případů o vlásek, '
        + 'takže to čtěte spolu s IQM (které ukazuje velikost rozdílu). Vyžaduje aspoň dva algoritmy '
        + 'spuštěné na sdílené hře.',
    },
    recommended: {
      en: 'A confident claim needs the value clearly off 0.5 AND its interval not crossing 0.5. Pair it with '
        + 'IQM for the size of the difference.',
      cz: 'Sebevědomé tvrzení potřebuje hodnotu jasně mimo 0,5 A zároveň interval, který 0,5 nepřekračuje. '
        + 'Doplňte ho o IQM pro velikost rozdílu.',
    },
  },
  analysis_export: {
    general: {
      en: 'Download the current run selection as a **citable dataset** — computed server-side from the full '
        + 'on-disk history, so exports are full-resolution, not the trimmed live view.\n'
        + '**CSV (tidy):** one row per (run, point, metric) — the universal format for pandas / R / Excel.\n'
        + '**Excel (.xlsx):** a publication workbook — a summary-stats sheet, a per-game/algorithm sheet '
        + 'with a native chart, plus config + methods.\n'
        + '**Repro card:** a Markdown card with a config hash, a BibTeX entry and the exact command to '
        + 'reproduce the run.\n'
        + '**LaTeX table:** a paste-ready booktabs results table.\n'
        + '**Vector figure (SVG):** a standalone line chart of the selected curves — drop it straight into a '
        + 'paper or slides; scales crisply at any size.\n'
        + '**TensorBoard:** a .zip of event files, one log folder per run — unzip it and run '
        + '`tensorboard --logdir` to browse the curves interactively.\n'
        + 'CSV, Excel and the figure follow the **compare mode** — per-game gives raw reward, per-algorithm '
        + 'gives the normalized skill-%.',
      cz: 'Stáhni aktuální výběr běhů jako **citovatelný dataset** — počítá se na serveru z celé historie na '
        + 'disku, takže exporty jsou v plném rozlišení, ne oříznutý živý pohled.\n'
        + '**CSV (tidy):** řádek na (běh, bod, metrika) — univerzální formát pro pandas / R / Excel.\n'
        + '**Excel (.xlsx):** publikační sešit — list souhrnných statistik, list na hru/algoritmus s '
        + 'nativním grafem, plus konfigurace + metody.\n'
        + '**Repro karta:** Markdown karta s hashem konfigurace, záznamem BibTeX a přesným příkazem k '
        + 'reprodukci běhu.\n'
        + '**LaTeX tabulka:** hotová booktabs tabulka výsledků.\n'
        + '**Vektorový obrázek (SVG):** samostatný čárový graf vybraných křivek — vložíte ho rovnou do článku '
        + 'nebo prezentace; ostrý v jakékoli velikosti.\n'
        + '**TensorBoard:** .zip se soubory událostí, jedna složka na běh — rozbalte a spusťte '
        + '`tensorboard --logdir` pro interaktivní prohlížení křivek.\n'
        + 'CSV, Excel i obrázek se řídí **režimem porovnání** — podle hry dá surovou odměnu, podle algoritmu '
        + 'normalizované skill-%.',
    },
  },
  // CPU/GPU training badge (parked C2 diagnostic, 2026-06-18): explains the gate-vs-device nuance the
  // GPU-utilization probe surfaced — a "GPU game" with an idle GPU panel is not a bug.
  training_device: {
    general: {
      en: '**Where this run trains — CPU or GPU.** It depends on what the agent looks at, not on how '
        + 'powerful your computer is.\n'
        + '**Picture games (Atari, ViZDoom)** learn straight from the screen pixels with a **convolutional** network. '
        + 'That is heavy maths the graphics card does best, so these train on the **GPU**.\n'
        + '**Board games with AlphaZero** also train on the **GPU**: AlphaZero plays many self-play games '
        + 'at once and learns with a bigger network, and that batched work is exactly what a graphics card '
        + 'is fast at. (The same board game under PPO uses a tiny network and trains on the CPU.)\n'
        + '**Everything else** (CartPole, LunarLander, the robots…) reads a short list of '
        + 'numbers, so its network is tiny. A tiny network is actually **faster on the CPU** — shipping '
        + 'such small batches over to the GPU costs more than just computing them (measured here: about '
        + '**3× faster on the CPU**). So these train on the CPU even on a GPU machine.\n'
        + '**Why is the GPU panel low then?** For the picture games the speed limit is the game emulators '
        + 'stepping on the CPU, and the network is small for such a strong card — so a modest GPU reading '
        + 'is normal, not a fault. A few games are marked "needs GPU" only because they need millions of '
        + 'steps (a time budget), not because they run on the GPU.',
      // RichText supports only **bold** and \n — no single-* italics (they would render literally).
      cz: '**Kde tento běh trénuje — na CPU, nebo na GPU.** Záleží na tom, na co se agent dívá, ne na tom, '
        + 'jak výkonný máte počítač.\n'
        + '**Obrázkové hry (Atari, ViZDoom)** se učí přímo z pixelů obrazovky pomocí **konvoluční** sítě. To je náročná '
        + 'matematika, kterou nejlépe zvládne grafická karta, takže tyto trénují na **GPU**.\n'
        + '**Deskové hry s AlphaZero** trénují také na **GPU**: AlphaZero hraje mnoho self-play partií '
        + 'najednou a učí se s větší sítí, a přesně taková dávková práce grafické kartě sedí. (Stejná desková '
        + 'hra s PPO používá drobnou síť a trénuje na CPU.)\n'
        + '**Všechno ostatní** (CartPole, LunarLander, roboti…) čte krátký seznam čísel, takže '
        + 'jeho síť je drobná. Drobná síť je ve skutečnosti **rychlejší na CPU** — posílat tak malé dávky na '
        + 'GPU stojí víc než je rovnou spočítat (zde naměřeno: zhruba **3× rychleji na CPU**). Proto tyto '
        + 'trénují na CPU i na stroji s GPU.\n'
        + '**Proč je tedy GPU panel nízko?** U obrázkových her je rychlostním stropem krokování herních '
        + 'emulátorů na CPU a síť je na tak silnou kartu malá — takže mírné vytížení GPU je normální, ne '
        + 'chyba. Pár her je označených „potřebuje GPU“ jen proto, že potřebují miliony kroků (časový '
        + 'rozpočet), ne proto, že by běžely na GPU.',
    },
  },
  // Hardware monitor panel (G4b) — a read-only explainer for the live CPU/GPU readings floated on the
  // chart during a run. Metric-only (no recommended/range/perEnv), like training_device above.
  hw_stats: {
    general: {
      en: '**Live hardware readings while a run trains.** They show how hard your computer is working — '
        + 'useful to confirm the GPU is actually busy on a GPU game, or just out of curiosity.\n'
        + '**CPU · Util** — how hard this app is working the processor. It can climb past 100% because a '
        + 'modern CPU has many cores, so "150%" just means it is using about one and a half cores.\n'
        + '**CPU · RAM** — ordinary memory in use / total, in gigabytes.\n'
        + '**GPU · Util** — how busy the graphics card is. High during picture-game (Atari) and AlphaZero '
        + 'training; low or hidden otherwise (most games train on the CPU — see the training-device note).\n'
        + '**GPU · VRAM** — the graphics card\'s own memory in use / total. A big model uses more.\n'
        + '**GPU · Temp / Power** — how hot the card is and how many watts it is drawing right now.\n'
        + 'The **GPU column appears only when the run actually trains on the graphics card**; a CPU run '
        + 'hides it rather than showing an idle card. Any reading that is unavailable shows "—".',
      cz: '**Živé údaje o hardwaru během tréninku.** Ukazují, jak moc váš počítač pracuje — hodí se ověřit, '
        + 'že GPU u obrázkové hry opravdu pracuje, nebo jen ze zvědavosti.\n'
        + '**CPU · Util** — jak moc tato aplikace zatěžuje procesor. Může přesáhnout 100 %, protože moderní '
        + 'procesor má mnoho jader, takže „150 %“ znamená využití zhruba jednoho a půl jádra.\n'
        + '**CPU · RAM** — běžná paměť používaná / celkem, v gigabajtech.\n'
        + '**GPU · Util** — jak je grafická karta vytížená. Vysoko při tréninku obrázkových her (Atari) a '
        + 'AlphaZero; jinak nízko nebo skrytá (většina her trénuje na CPU — viz poznámka o zařízení tréninku).\n'
        + '**GPU · VRAM** — vlastní paměť grafické karty používaná / celkem. Větší model spotřebuje víc.\n'
        + '**GPU · Temp / Power** — jak je karta horká a kolik wattů právě odebírá.\n'
        + 'Sloupec **GPU se zobrazí jen tehdy, když běh skutečně trénuje na grafické kartě**; běh na CPU ho '
        + 'skryje, místo aby ukazoval nečinnou kartu. Jakýkoli nedostupný údaj se zobrazí jako „—“.',
    },
  },
  // Board games (G6a): the opponent's strength. Not a training hyperparameter — a play-time choice —
  // but it ships the same info popup as every tunable (general + per-game + a ★ recommended value).
  board_difficulty: {
    general: {
      en: '**How strong the AI opponent plays.** The built-in AI is a *Monte-Carlo Tree Search* (MCTS): '
        + 'before each move it plays out many quick random games in its head and picks the move that wins '
        + 'most often. Difficulty = how many of those look-ahead games it runs — **Easy** runs only a few '
        + '(it misses things, so a beginner can win), **Hard** runs a lot (it plays essentially perfectly). '
        + 'It learns nothing between games; it just searches harder. (A *trained* neural opponent arrives '
        + 'in a later step.)',
      cz: '**Jak silně hraje soupeřící AI.** Vestavěná AI je *Monte-Carlo stromové prohledávání* (MCTS): '
        + 'před každým tahem si v hlavě rychle přehraje mnoho náhodných partií a vybere tah, který vyhrává '
        + 'nejčastěji. Obtížnost = kolik takových partií dopředu si přehraje — **Lehká** jen pár (přehlédne '
        + 'věci, takže začátečník může vyhrát), **Těžká** jich přehraje hodně (hraje prakticky dokonale). '
        + 'Mezi partiemi se nic neučí, jen hledá důkladněji. (*Natrénovaný* neuronový soupeř přijde v '
        + 'dalším kroku.)',
    },
    recommended: {
      en: 'Start on **Medium** to get a feel for the game, drop to Easy for a confidence win, and try Hard '
        + 'once you want a real test — against perfect play the best you can do is a draw.',
      cz: 'Začněte na **Střední**, ať hru pochytíte, přepněte na Lehkou pro jisté vítězství a Těžkou '
        + 'zkuste, až budete chtít opravdovou výzvu — proti dokonalé hře je nejlepší možný výsledek remíza.',
    },
    perEnv: {
      tictactoe: {
        en: 'Tic-Tac-Toe is fully "solved": with no mistakes every game ends in a draw. So on **Hard** the '
          + 'AI never loses — your goal is to not lose either (force the draw). On **Easy** it searches so '
          + 'little that it will sometimes hand you the win.',
        cz: 'Piškvorky 3×3 jsou zcela „vyřešené“: bez chyb končí každá partie remízou. Na **Těžké** tedy AI '
          + 'nikdy neprohraje — vaším cílem je také neprohrát (vynutit remízu). Na **Lehké** prohledává tak '
          + 'málo, že vám občas vítězství daruje.',
      },
      connect_four: {
        en: 'Connect Four is a much bigger game than Tic-Tac-Toe, so difficulty matters more here. On '
          + '**Easy** the AI searches only a little and makes real mistakes you can punish. On **Hard** it '
          + 'looks much further ahead and rarely slips — expect a genuine fight. **Medium** is a balanced '
          + 'first test.',
        cz: 'Čtyři v řadě jsou mnohem větší hra než piškvorky, takže obtížnost tu hraje větší roli. Na '
          + '**Lehké** AI prohledává jen málo a dělá skutečné chyby, které můžete potrestat. Na **Těžké** '
          + 'vidí mnohem dál a chybuje zřídka — čekejte opravdový souboj. **Střední** je vyvážená první '
          + 'zkouška.',
      },
      othello: {
        en: 'Othello is the biggest board game here, with constant swings, so look-ahead pays off and '
          + 'difficulty really bites. On **Easy** the AI searches little and misjudges the late flips you '
          + 'can exploit. On **Hard** it sees the reversals coming and plays for corners — a stiff test. '
          + '**Medium** is a fair first game.',
        cz: 'Othello je tu největší desková hra plná neustálých zvratů, takže prohledávání dopředu se '
          + 'vyplácí a obtížnost je hodně znát. Na **Lehké** AI prohledává málo a špatně odhaduje pozdní '
          + 'obraty, kterých můžete využít. Na **Těžké** zvraty předvídá a hraje na rohy — tvrdá zkouška. '
          + '**Střední** je férová první partie.',
      },
      breakthrough: {
        en: 'Breakthrough rewards looking a few moves ahead — one careless advance can hand the AI a '
          + 'capture that opens a lane. On **Easy** the AI searches little and lets pieces run past '
          + 'undefended. On **Hard** it spots your breakthroughs early and blocks them — a real fight. '
          + '**Medium** is a fair first game.',
        cz: 'Breakthrough odměňuje výhled o pár tahů dopředu — jeden neopatrný postup může AI darovat '
          + 'sebrání, které otevře cestu. Na **Lehké** AI prohledává málo a nechá figurky proběhnout bez '
          + 'obrany. Na **Těžké** vaše průlomy odhalí včas a zablokuje je — opravdový souboj. **Střední** '
          + 'je férová první partie.',
      },
    },
  },
  algorithm: {
    general: {
      en: '**How the agent learns.**\n**PPO** (reinforcement learning) — tweaks one neural network with gradients after each batch of play; steady and sample-efficient.\n**Neuroevolution** — keeps a whole population of networks, scores them, and breeds the best (mutation + crossover) each generation; simple, gradient-free, like "survival of the fittest".\n**Q-learning** — builds a plain table of "how good is each action in each state" and refines it from experience; the classic value-based method, available on the small grid-world games where you can literally watch the table fill in.\n**DQN** (Deep Q-Network, button-based games) — the **value-based** counterpart to PPO and the original deep-RL breakthrough (Atari, 2015). It is Q-learning with a *neural network* instead of a table, so it scales to games too big to tabulate; like SAC/TD3 it is **off-policy** (a replay buffer) and it explores by acting at random a shrinking fraction of the time (ε-greedy). The headline comparison: PPO (policy) vs DQN (value) on the very same game.\n**AlphaZero** (board games only) — a network learns purely by **playing itself**, using look-ahead search (the same kind the built-in AI uses) to pick strong moves and then training on them. No human examples — it bootstraps from nothing. The famous recipe behind superhuman chess/Go engines, scaled down here.\n**SAC** (Soft Actor-Critic, continuous-control games only) — an **off-policy** method that stores past experience in a replay buffer and reuses it, so it learns from far fewer steps than PPO; it also rewards staying a little unpredictable (entropy), which helps it explore. The right tool for the smooth joint-torque robots — it actually solves the hardest of them.\n**TD3** (Twin Delayed DDPG, continuous-control games only) — SAC\'s **off-policy** sibling, just as sample-efficient. Instead of the entropy trick its policy is *deterministic* (one action per state) and it explores by adding a little noise to its actions; its name comes from its two stability tricks (twin value networks + delayed updates). A great second method to put head-to-head against SAC.\n**A2C** (Advantage Actor-Critic, button- and stick-based games) — PPO\'s simpler predecessor and its on-policy sibling: the same actor-critic idea, but it updates after only a handful of steps with a single plain gradient step (no clipping, no replay buffer). It usually learns a little slower and noisier than PPO — which is exactly what makes it the clearest **PPO-vs-A2C** lesson: same family, so the gap shows you what PPO\'s clipping and rollout reuse actually buy.\n**QR-DQN** (Quantile-Regression DQN, button-based games) — DQN made *distributional*. Where DQN learns a single average score for each action, QR-DQN learns the whole spread of possible outcomes (a set of quantiles) and acts on their average. Same off-policy, ε-greedy machinery as DQN, offered on the same games — so it is the clean **DQN-vs-QR-DQN** lesson: any difference is down to learning the full distribution rather than just the mean (the idea behind the "Rainbow" agent, and historically a strong performer on Atari).',
      cz: '**Jak se agent učí.**\n**PPO** (zpětnovazební učení) — upravuje jednu neuronovou síť pomocí gradientů po každé dávce hraní; stabilní a úsporné na data.\n**Neuroevoluce** — udržuje celou populaci sítí, ohodnotí je a v každé generaci množí ty nejlepší (mutace + křížení); jednoduchá, bez gradientů, jako „přežití nejschopnějších“.\n**Q-učení** — sestavuje jednoduchou tabulku „jak dobrá je každá akce v každém stavu“ a vylepšuje ji ze zkušenosti; klasická hodnotová metoda, dostupná u malých mřížkových her, kde můžete doslova sledovat, jak se tabulka plní.\n**DQN** (Deep Q-Network, hry s tlačítky) — **hodnotový** protějšek PPO a původní průlom hlubokého RL (Atari, 2015). Je to Q-učení s *neuronovou sítí* místo tabulky, takže škáluje i na hry příliš velké na tabulku; stejně jako SAC/TD3 je **off-policy** (replay buffer) a zkoumá tím, že zmenšující se část času jedná náhodně (ε-greedy). Hlavní srovnání: PPO (strategie) vs DQN (hodnota) na úplně stejné hře.\n**AlphaZero** (jen deskové hry) — síť se učí výhradně **hrou sama proti sobě**; k výběru silných tahů používá prohledávání dopředu (stejného druhu jako vestavěná AI) a pak se na nich učí. Žádné lidské příklady — startuje od nuly. Slavný recept za nadlidskými šachovými/go enginy, tady ve zmenšené podobě.\n**SAC** (Soft Actor-Critic, jen hry se spojitým řízením) — **off-policy** metoda, která ukládá minulé zkušenosti do paměti (replay buffer) a znovu je využívá, takže se učí z mnohem méně kroků než PPO; navíc odměňuje trochu nepředvídatelné chování (entropii), což pomáhá zkoumat. Správný nástroj pro roboty s plynulými momenty v kloubech — ty nejtěžší skutečně vyřeší.\n**TD3** (Twin Delayed DDPG, jen hry se spojitým řízením) — **off-policy** sourozenec SAC, stejně úsporný na data. Místo triku s entropií je jeho strategie *deterministická* (jedna akce na stav) a zkoumá tak, že ke svým akcím přidává trochu šumu; název má podle svých dvou triků pro stabilitu (dvojice hodnotových sítí + zpožděné aktualizace). Skvělá druhá metoda do souboje proti SAC.\n**A2C** (Advantage Actor-Critic, hry s tlačítky i pákou) — jednodušší předchůdce PPO a jeho on-policy sourozenec: stejná myšlenka actor-critic, ale aktualizuje se už po pár krocích jediným prostým gradientovým krokem (žádné ořezávání, žádný replay buffer). Obvykle se učí o něco pomaleji a rozkolísaněji než PPO — a právě proto je to nejjasnější lekce **PPO vs A2C**: stejná rodina, takže rozdíl ukazuje, co vám ořezávání a znovupoužití rolloutu v PPO reálně přinášejí.\n**QR-DQN** (Quantile-Regression DQN, hry s tlačítky) — distribuční verze DQN. Zatímco DQN se učí jediné průměrné skóre pro každou akci, QR-DQN se učí celé rozložení možných výsledků (sadu kvantilů) a jedná podle jejich průměru. Stejná off-policy, ε-greedy mašinerie jako DQN, nabízené na stejných hrách — takže je to čistá lekce **DQN vs QR-DQN**: jakýkoli rozdíl plyne z učení celé distribuce místo pouhého průměru (myšlenka za agentem „Rainbow“ a historicky silný hráč na Atari).',
    },
    recommended: {
      en: 'PPO — the reliable, general-purpose default that also scales to harder games. Neuroevolution is gradient-free and can be surprisingly fast on simple tasks. On the grid-world games, Q-learning is the star. On the **board games** you also get **AlphaZero**, which learns by playing itself and searches ahead while playing — a fun head-to-head against PPO on the very same game. On the **continuous-control games** (the robots, BipedalWalker, Pendulum) you also get **SAC** and **TD3**, two off-policy methods far more sample-efficient than PPO and the ones that really shine there — a great three-way comparison on one task. Where it is offered (CartPole and the other classic-control games), **A2C** — PPO\'s simpler on-policy predecessor — is worth adding to the line-up to see, side by side, what PPO\'s clipping and rollout reuse buy over the plain actor-critic. Try them and compare (see the per-game note below).',
      cz: 'PPO — spolehlivá, univerzální volba, která zvládne i těžší hry. Neuroevoluce je bezgradientní a u jednoduchých úloh bývá překvapivě rychlá. U mřížkových her je hvězdou Q-učení. U **deskových her** máte navíc **AlphaZero**, který se učí hrou sám proti sobě a při hře prohledává dopředu — pěkný souboj s PPO na úplně stejné hře. U **her se spojitým řízením** (roboti, BipedalWalker, Pendulum) máte navíc **SAC** a **TD3**, dvě off-policy metody mnohem úspornější na data než PPO, které tam opravdu vynikají — skvělé třístranné srovnání na jedné úloze. Kde je k dispozici (CartPole a další hry klasického řízení), stojí za to přidat do sestavy i **A2C** — jednodušší on-policy předchůdce PPO — a vedle sebe vidět, co vám ořezávání a znovupoužití rolloutu v PPO dávají navíc oproti prostému actor-critic. Vyzkoušejte a porovnejte (viz poznámka k dané hře níže).',
    },
    perEnv: {
      cartpole: {
        en: 'The best place to compare all **five**. **PPO** (policy-gradient) and **DQN** (value-based) both solve CartPole — this is *the* classic head-to-head, the two great families of deep RL on one easy task: watch DQN\'s curve and PPO\'s climb to 500 and overlay them in run-compare. **A2C** is PPO\'s simpler on-policy predecessor: it climbs steadily but on a single environment it is noisier and usually lands well short of PPO\'s clean 500 in the same budget — overlay the two to *see* what PPO\'s clipping and rollout reuse actually buy. **QR-DQN** is DQN made distributional (it learns the whole spread of returns, not just the mean) — overlay it on DQN to see whether the distribution helps on this easy task. **Neuroevolution** is often the fastest here (a tiny network, no gradients, just scoring a population). DQN needs a bit more budget than PPO on this trivial task (hence its larger ★ steps), but it learns reliably.',
        cz: 'Nejlepší místo pro porovnání všech **pěti**. **PPO** (policy-gradient) a **DQN** (hodnotové) CartPole vyřeší obě — tohle je *ten* klasický souboj, dvě velké rodiny hlubokého RL na jedné snadné úloze: sledujte, jak křivka DQN i PPO vyšplhá k 500, a porovnejte je v překrytí běhů. **A2C** je jednodušší on-policy předchůdce PPO: plynule stoupá, ale na jednom prostředí je rozkolísanější a při stejném rozpočtu obvykle zůstane výrazně pod čistou 500 od PPO — překryjte oba a *uvidíte*, co ořezávání a znovupoužití rolloutu v PPO reálně přinášejí. **QR-DQN** je distribuční verze DQN (učí se celé rozložení návratů, ne jen průměr) — překryjte ji s DQN a uvidíte, jestli distribuce na téhle snadné úloze pomáhá. **Neuroevoluce** tu bývá nejrychlejší (drobná síť, žádné gradienty, jen ohodnocení populace). DQN potřebuje na téhle triviální úloze o něco větší rozpočet než PPO (proto jeho větší ★ kroky), ale učí se spolehlivě.',
      },
      lunarlander: {
        en: 'PPO and DQN are both strong here, Neuroevolution less so. **PPO** with enough steps (a few hundred thousand) reliably learns to land. **DQN** (value-based, off-policy) is a great fit for this four-button game and is often very sample-efficient — a good second method to race against PPO. The simple neuroevolution uses a tiny network, so it improves but rarely reaches a clean landing within a few dozen generations. **A2C** (PPO\'s simpler on-policy predecessor) also learns to land given enough steps, usually a bit slower than PPO — a good fourth method to compare. **QR-DQN** (distributional DQN) is a natural fifth — race it against plain DQN to see if learning the full return distribution pays off here.',
        cz: 'PPO i DQN jsou tu silné, Neuroevoluce méně. **PPO** se s dostatkem kroků (řádově statisíce) spolehlivě naučí přistávat. **DQN** (hodnotové, off-policy) téhle hře se čtyřmi tlačítky dobře sedí a bývá velmi úsporné na data — dobrá druhá metoda do souboje s PPO. Zdejší jednoduchá neuroevoluce používá drobnou síť, takže se sice zlepšuje, ale za pár desítek generací málokdy dosáhne čistého přistání. **A2C** (jednodušší on-policy předchůdce PPO) se s dostatkem kroků přistávat také naučí, obvykle o něco pomaleji než PPO — dobrá čtvrtá metoda k porovnání. **QR-DQN** (distribuční DQN) je přirozená pátá — postavte ji proti obyčejnému DQN a uvidíte, jestli se učení celého rozložení návratů tady vyplatí.',
      },
      mountaincar: {
        en: 'A famous exploration trap, hard for every method here. **PPO** with the default settings often stalls near −200: a random agent almost never reaches the flag, so there is no reward signal (raising the Entropy bonus helps). **DQN** explores by ε-greedy and its replay buffer remembers the rare flag-reaching runs, so the tuned recipe can break the trap — but it is genuinely hard for value-based learning too. **Neuroevolution** can do better — among a whole population some network stumbles onto the rocking motion. **A2C** (on-policy, like PPO) faces the same reward-signal problem and often needs the entropy nudge too. **QR-DQN** (distributional DQN) explores exactly like DQN, so it hits the same trap — a fair fifth to compare. A great task for comparing all five.',
        cz: 'Pověstná past na zkoumání, těžká tu pro každou metodu. **PPO** s výchozím nastavením často uvázne kolem −200: náhodný agent skoro nikdy nedojede k vlajce, takže není z čeho se učit (pomáhá zvýšit bonus za entropii). **DQN** zkoumá pomocí ε-greedy a jeho replay buffer si pamatuje vzácné úspěšné jízdy k vlajce, takže laděný recept past prolomit umí — ale i pro hodnotové učení je to vážně těžké. **Neuroevoluce** tu může být lepší — v celé populaci některá síť náhodou objeví houpavý pohyb. **A2C** (on-policy jako PPO) čelí témuž problému s odměnou a často také potřebuje přidat entropii. **QR-DQN** (distribuční DQN) zkoumá stejně jako DQN, takže naráží na tutéž past — férová pátá do porovnání. Skvělá úloha pro porovnání všech pěti.',
      },
      acrobot: {
        en: 'All three handle Acrobot well. **PPO** reliably learns the pumping motion within a few hundred thousand steps and reaches the target (around −100). **DQN** (value-based, off-policy) also learns it efficiently — another clean PPO-vs-DQN comparison. **Neuroevolution** climbs steadily too, since even a tiny network can discover the rhythmic swing. **A2C** (PPO\'s simpler on-policy predecessor) also learns the swing — usually a bit noisier than PPO, a nice fourth curve to overlay. **QR-DQN** (distributional DQN) learns it efficiently too — a clean fifth for a DQN-vs-QR-DQN overlay.',
        cz: 'Acrobot zvládnou všechny. **PPO** se za pár set tisíc kroků spolehlivě naučí „pumpovat“ a dosáhne cíle (kolem −100). **DQN** (hodnotové, off-policy) se ho také naučí úsporně — další čisté srovnání PPO vs DQN. **Neuroevoluce** se tu rovněž plynule zlepšuje, protože i drobná síť objeví rytmické houpání. **A2C** (jednodušší on-policy předchůdce PPO) se houpání také naučí — obvykle rozkolísaněji než PPO, pěkná čtvrtá křivka do překrytí. **QR-DQN** (distribuční DQN) se ho naučí také úsporně — čistá pátá pro překrytí DQN vs QR-DQN.',
      },
      pendulum: {
        en: 'Four methods work on Pendulum. **SAC** and **TD3** are the standouts — both off-policy and very sample-efficient, they nail the swing-up-and-hold in a fraction of the steps PPO needs (a perfect first place to feel the off-policy-vs-PPO difference, and to compare SAC vs TD3 side by side). **PPO** is the steady on-policy baseline. **Neuroevolution** also works (its networks output a continuous torque) and is worth comparing. **A2C** (on-policy, like PPO) also handles the continuous torque, but as an on-policy method it is far less sample-efficient than SAC/TD3 here — a good illustration of why off-policy wins on continuous control.',
        cz: 'Na Pendulu fungují čtyři metody. **SAC** a **TD3** vynikají — obě jsou off-policy a velmi úsporné na data, takže vyhoupnutí a udržení zvládnou ve zlomku kroků oproti PPO (ideální místo, kde pocítíte rozdíl off-policy vs PPO a kde porovnáte SAC a TD3 vedle sebe). **PPO** je stabilní on-policy základ. **Neuroevoluce** také funguje (její sítě vydávají spojitý moment) a stojí za porovnání. **A2C** (on-policy jako PPO) si se spojitým momentem také poradí, ale jako on-policy metoda je tu mnohem méně úsporná na data než SAC/TD3 — pěkná ukázka, proč na spojitém řízení vítězí off-policy.',
      },
      mountaincarcontinuous: {
        en: 'A continuous exploration trap. **PPO** often stalls near 0: it rarely stumbles onto the flag, and the small force penalty discourages trying. **Neuroevolution** tends to do better — among a whole population some network reaches the flag and that success spreads. **SAC** explores via its entropy bonus and its replay buffer remembers the rare flag-reaching runs, so it can break the trap too; **TD3** explores via its injected action noise and the same replay memory. **A2C** (on-policy, like PPO) tends to stall here for the same reason PPO does — the off-policy methods\' replay memory of rare successes is the real edge. A great task for comparing them all.',
        cz: 'Spojitá past na zkoumání. **PPO** často uvázne poblíž 0: na vlajku jen zřídka náhodou narazí a malá penalizace za sílu odrazuje od zkoušení. **Neuroevoluce** tu bývá lepší — v celé populaci některá síť dojede k vlajce a úspěch se rozšíří. **SAC** zkoumá díky bonusu za entropii a jeho replay buffer si pamatuje vzácné úspěšné jízdy k vlajce, takže past také prolomí; **TD3** zkoumá díky přidanému šumu do akcí a stejné paměti přehrávání. **A2C** (on-policy jako PPO) tu ze stejného důvodu jako PPO často uvázne — skutečnou výhodou je paměť vzácných úspěchů u off-policy metod. Skvělá úloha pro porovnání všech.',
      },
      bipedalwalker: {
        en: 'PPO, SAC and TD3 are offered here. Walking is a hard continuous-control task with four leg joints. **PPO** learns a gait given a few million steps; **SAC** and **TD3** (both off-policy) are usually more sample-efficient and strong choices for this kind of locomotion — a good place to race the two off-policy methods against each other. Neuroevolution is deliberately turned off — population search is impractical here. Training is reserved for a GPU machine; you can still play it by hand now and watch a trained AI.',
        cz: 'Tady jsou k dispozici PPO, SAC a TD3. Chůze je těžká úloha spojitého řízení se čtyřmi klouby nohou. **PPO** se chůzi naučí za pár milionů kroků; **SAC** a **TD3** (obě off-policy) bývají úspornější na data a jsou pro tento druh pohybu silnou volbou — dobré místo, kde proti sobě postavit obě off-policy metody. Neuroevoluce je záměrně vypnutá — populační hledání je tu nepraktické. Trénink je vyhrazen pro stroj s GPU; rukama si to ale zahrajete už teď a natrénovanou AI můžete sledovat.',
      },
      bipedalwalkerhardcore: {
        en: 'PPO, SAC and TD3, same as the standard course — and the hardcore terrain (ladders, stumps, pits) makes it one of the hardest continuous-control benchmarks, needing the largest budget. The off-policy methods\' (SAC / TD3) sample efficiency helps here. Training is GPU-only; play it by hand now.',
        cz: 'PPO, SAC a TD3, stejně jako u standardní dráhy — a hardcore terén (žebříky, pařezy, jámy) z ní dělá jeden z nejtěžších benchmarků spojitého řízení, který potřebuje největší rozpočet. Úspornost off-policy metod (SAC / TD3) tu pomáhá. Trénink je jen na GPU; rukama si to zahrajete už teď.',
      },
      carracing: {
        en: 'Only PPO is offered here: on a GPU it trains a convolutional network (CnnPolicy) straight from the 96×96 picture — a different policy from the small games\' MLP. Neuroevolution is turned off as data (a flat-vector genome cannot take pixels). Training needs a CUDA GPU, so Run is enabled on a GPU machine (and stays disabled on a CPU-only box); you can also play it by hand and watch your trained AI drive.',
        cz: 'Tady je k dispozici jen PPO: na GPU trénuje konvoluční síť (CnnPolicy) přímo z obrazu 96×96 — jiná strategie než MLP u malých her. Neuroevoluce je vypnutá jako data (genom z plochého vektoru neumí pixely). Trénink potřebuje GPU (CUDA), takže Spustit je povolené na stroji s GPU (a na čistém CPU zůstává zakázané); můžete si to i zahrát rukama a sledovat, jak vaše natrénovaná AI řídí.',
      },
      doom_basic: {
        en: 'Three image algorithms are offered — **PPO**, **DQN** and **QR-DQN** — each training a convolutional network (CnnPolicy) on a GPU straight from the 3D view. Basic is VizDoom\'s "hello world": the agent only has to strafe and shoot one monster, so it learns fast — PPO reaches a positive score within tens of thousands of steps, and DQN (value-based) with its distributional cousin QR-DQN both learn it too, making a clean image-based head-to-head. Neuroevolution and Q-learning are off as data (a flat genome / a lookup table cannot take pixels). Training needs a CUDA GPU, so Run is enabled only on a GPU machine; you can play it by hand and watch your trained AI on any machine.',
        cz: 'K dispozici jsou tři obrázkové algoritmy — **PPO**, **DQN** a **QR-DQN** — každý trénuje na GPU konvoluční síť (CnnPolicy) přímo z 3D pohledu. Basic je „hello world“ VizDoomu: agent se musí jen uhýbat do stran a zastřelit jednu nestvůru, takže se učí rychle — PPO se za desítky tisíc kroků dostane do kladného skóre a DQN (hodnotové) i jeho distribuční příbuzný QR-DQN se ho také naučí, což dává čisté obrázkové srovnání. Neuroevoluce a Q-učení jsou vypnuté jako data (plochý genom ani tabulka neumí pixely). Trénink potřebuje GPU (CUDA), takže Spustit je povolené jen na stroji s GPU; rukama si to zahrajete a natrénovanou AI můžete sledovat na jakémkoli stroji.',
      },
      doom_defend_center: {
        en: 'Same three image algorithms as the other VizDoom scenarios — **PPO**, **DQN**, **QR-DQN** (CnnPolicy on a GPU). Here the agent is fixed in the centre and must turn and shoot the monsters closing in, with limited ammo — harder than Basic, so it needs more steps to climb. PPO is a reliable first choice; race DQN and QR-DQN against it to compare the value-based family. Neuroevolution and Q-learning are off (pixels). Training is GPU-only; play by hand and watch a trained AI anywhere.',
        cz: 'Stejné tři obrázkové algoritmy jako u ostatních scénářů VizDoomu — **PPO**, **DQN**, **QR-DQN** (CnnPolicy na GPU). Tady je agent připevněný uprostřed a musí se otáčet a střílet blížící se nestvůry s omezeným střelivem — těžší než Basic, takže potřebuje víc kroků, než se rozjede. PPO je spolehlivá první volba; postavte proti němu DQN a QR-DQN a porovnejte hodnotovou rodinu. Neuroevoluce a Q-učení jsou vypnuté (pixely). Trénink je jen na GPU; rukama si zahrajete a natrénovanou AI můžete sledovat kdekoli.',
      },
      doom_health_gathering: {
        en: 'The same three image algorithms — **PPO**, **DQN**, **QR-DQN** (CnnPolicy on a GPU). This one adds real 3D navigation: the acid floor drains health, so the agent must walk around and collect medkits to survive — the hardest of the three scenarios, needing the most steps. PPO handles the navigation well; DQN and QR-DQN are worth comparing. Neuroevolution and Q-learning are off (pixels). Training is GPU-only; play by hand and watch a trained AI anywhere.',
        cz: 'Stejné tři obrázkové algoritmy — **PPO**, **DQN**, **QR-DQN** (CnnPolicy na GPU). Tenhle přidává skutečnou 3D navigaci: kyselá podlaha ubírá zdraví, takže agent musí chodit po místnosti a sbírat lékárničky, aby přežil — nejtěžší ze tří scénářů, potřebuje nejvíc kroků. PPO navigaci zvládá dobře; DQN a QR-DQN stojí za porovnání. Neuroevoluce a Q-učení jsou vypnuté (pixely). Trénink je jen na GPU; rukama si zahrajete a natrénovanou AI můžete sledovat kdekoli.',
      },
      doom_defend_line: {
        en: 'Same three image algorithms as the other VizDoom scenarios — **PPO**, **DQN**, **QR-DQN** (CnnPolicy on a GPU). A close cousin of Defend the Center: the agent is rooted in place and must turn and shoot a line of monsters advancing from the front, with limited ammo. PPO is a reliable first choice; race DQN and QR-DQN against it to compare the value-based family. Neuroevolution and Q-learning are off (pixels). Training is GPU-only; play by hand and watch a trained AI anywhere.',
        cz: 'Stejné tři obrázkové algoritmy jako u ostatních scénářů VizDoomu — **PPO**, **DQN**, **QR-DQN** (CnnPolicy na GPU). Blízký příbuzný Defend the Center: agent je připevněný na místě a musí se otáčet a střílet řadu nestvůr blížících se zepředu, s omezeným střelivem. PPO je spolehlivá první volba; postavte proti němu DQN a QR-DQN a porovnejte hodnotovou rodinu. Neuroevoluce a Q-učení jsou vypnuté (pixely). Trénink je jen na GPU; rukama si zahrajete a natrénovanou AI můžete sledovat kdekoli.',
      },
      doom_health_gathering_supreme: {
        en: 'The same three image algorithms — **PPO**, **DQN**, **QR-DQN** (CnnPolicy on a GPU). A harder twist on Health Gathering: the acid floor still drains health and medkits still refill it, but now walls block the way and blue poison vials punish a careless grab, so the agent must navigate and tell good from bad — the toughest VizDoom scenario here, needing the most steps. PPO handles the navigation well; DQN and QR-DQN are worth comparing. Neuroevolution and Q-learning are off (pixels). Training is GPU-only; play by hand and watch a trained AI anywhere.',
        cz: 'Stejné tři obrázkové algoritmy — **PPO**, **DQN**, **QR-DQN** (CnnPolicy na GPU). Těžší varianta Health Gathering: kyselá podlaha stále ubírá zdraví a lékárničky ho doplňují, ale teď cestu blokují stěny a modré lahvičky s jedem trestají neopatrné sebrání, takže agent musí navigovat a rozlišovat dobré od špatného — nejtěžší zdejší scénář VizDoomu, potřebuje nejvíc kroků. PPO navigaci zvládá dobře; DQN a QR-DQN stojí za porovnání. Neuroevoluce a Q-učení jsou vypnuté (pixely). Trénink je jen na GPU; rukama si zahrajete a natrénovanou AI můžete sledovat kdekoli.',
      },
      doom_predict_position: {
        en: 'The same three image algorithms — **PPO**, **DQN**, **QR-DQN** (CnnPolicy on a GPU). This is the family\'s hardest *credit-assignment* task: the reward is **sparse** — a single rocket that must lead a moving monster either lands (+1) or it doesn\'t, with only a tiny time penalty in between — so the agent gets almost no feedback until the shot resolves. That makes the small entropy bonus (the ★ ent_coef) especially important to keep it exploring different aim timings. PPO tends to learn the lead most reliably; DQN and QR-DQN are worth racing. Neuroevolution and Q-learning are off (pixels). Training is GPU-only; play by hand and watch a trained AI anywhere.',
        cz: 'Stejné tři obrázkové algoritmy — **PPO**, **DQN**, **QR-DQN** (CnnPolicy na GPU). Tohle je nejtěžší úloha rodiny na *přiřazení zásluhy*: odměna je **řídká** — jediná raketa, která musí předvídat pohyb nestvůry, buď zasáhne (+1), nebo ne, mezitím jen drobný časový postih — takže agent nedostává skoro žádnou zpětnou vazbu, dokud se výstřel nevyhodnotí. Proto je malý entropický bonus (★ ent_coef) obzvlášť důležitý, aby dál zkoušel různé načasování míření. PPO se předvídání obvykle naučí nejspolehlivěji; DQN a QR-DQN stojí za souboj. Neuroevoluce a Q-učení jsou vypnuté (pixely). Trénink je jen na GPU; rukama si zahrajete a natrénovanou AI můžete sledovat kdekoli.',
      },
      doom_take_cover: {
        en: 'The same three image algorithms — **PPO**, **DQN**, **QR-DQN** (CnnPolicy on a GPU). The simplest controls of the family — just two actions, strafe left or strafe right — with a **dense** reward (+1 for every tic survived), so the learning signal is smooth and the score climbs steadily as the agent dodges better. It still demands quick pattern-reading of the incoming fireballs. PPO learns to weave reliably; DQN and QR-DQN are good value-based comparisons. Neuroevolution and Q-learning are off (pixels). Training is GPU-only; play by hand and watch a trained AI anywhere.',
        cz: 'Stejné tři obrázkové algoritmy — **PPO**, **DQN**, **QR-DQN** (CnnPolicy na GPU). Nejjednodušší ovládání v rodině — jen dvě akce, úkrok doleva nebo doprava — s **hustou** odměnou (+1 za každý přežitý tik), takže signál k učení je plynulý a skóre stabilně roste, jak se agent lépe vyhýbá. Přesto vyžaduje rychlé čtení vzoru přilétajících ohnivých koulí. PPO se naučí spolehlivě kličkovat; DQN a QR-DQN jsou dobrá hodnotová srovnání. Neuroevoluce a Q-učení jsou vypnuté (pixely). Trénink je jen na GPU; rukama si zahrajete a natrénovanou AI můžete sledovat kdekoli.',
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
      mpe_spread: {
        en: 'PPO is the only option here, with a twist: this is a *multi-agent* world, so PPO trains one shared "brain" that drives all the agents at once (parameter sharing). Neuroevolution and Q-learning have no multi-agent path, so they are turned off. Watch the agents spread out to cover the targets as the shared policy improves.',
        cz: 'Tady je jedinou volbou PPO, ale s vychytávkou: jde o *více­agentní* svět, takže PPO trénuje jeden sdílený „mozek“, který řídí všechny agenty najednou (sdílení parametrů). Neuroevoluce ani Q-učení nemají více­agentní cestu, takže jsou vypnuté. Sledujte, jak se agenti rozprostírají a pokrývají cíle, jak se sdílená strategie zlepšuje.',
      },
      mpe_spread_swarm: {
        en: 'PPO only, same as the 3-agent version — one shared policy now drives a six-agent swarm. The bigger the swarm, the more there is to coordinate, but it is still the single shared brain learning for everyone.',
        cz: 'Jen PPO, stejně jako u verze se třemi agenty — jedna sdílená strategie teď řídí roj šesti agentů. Čím větší roj, tím víc koordinace, ale stále se učí jeden sdílený mozek za všechny.',
      },
      mpe_tag: {
        en: 'PPO only — but unlike Simple Spread the two species (predators vs. prey) have different observations and opposite rewards, so one shared brain will not do: each species trains its own shared policy, alternating self-play against a frozen copy of the other. Watch the predators learn to fan out and corner the faster prey while the prey learns to dodge and use the obstacles.',
        cz: 'Jen PPO — ale na rozdíl od Simple Spread mají oba druhy (predátoři vs. kořist) jiná pozorování a opačné odměny, takže jeden sdílený mozek nestačí: každý druh si trénuje vlastní sdílenou strategii střídavým self-play proti zmrazené kopii toho druhého. Sledujte, jak se predátoři učí rozprostřít se a zahnat rychlejší kořist do kouta, zatímco kořist se učí uhýbat a využívat překážky.',
      },
      mpe_tag_pack: {
        en: 'PPO only, same as the 3-vs-1 version — a six-predator pack and two prey, still two opposed species that each train their own shared policy by alternating self-play.',
        cz: 'Jen PPO, stejně jako verze 3 na 1 — šestičlenná smečka predátorů a dvě kořisti, stále dva protichůdné druhy, z nichž každý si trénuje vlastní sdílenou strategii střídavým self-play.',
      },
      pursuit: {
        en: 'PPO only — Pursuit is a *cooperative* swarm, so (like Simple Spread) PPO trains one shared "brain" that drives all eight pursuers at once (parameter sharing). Neuroevolution and Q-learning have no multi-agent path, so they are turned off. Watch the shared policy learn to surround the evaders together rather than chase them one by one.',
        cz: 'Jen PPO — Pursuit je *kooperativní* roj, takže (jako Simple Spread) PPO trénuje jeden sdílený „mozek“, který řídí všech osm pronásledovatelů najednou (sdílení parametrů). Neuroevoluce ani Q-učení nemají více­agentní cestu, takže jsou vypnuté. Sledujte, jak se sdílená strategie učí kořist společně obkličovat místo aby ji honila po jednom.',
      },
      multiwalker: {
        en: 'PPO only — Multiwalker is a *cooperative* swarm with continuous control, so (like Pursuit) PPO trains one shared "brain" that drives all three walkers at once (parameter sharing). Neuroevolution and Q-learning have no multi-agent path, so they are turned off. A heads-up on difficulty vs. algorithm: this task is genuinely hard for PPO. It reliably learns the *easy* half — to stop falling — but learning to actually walk the package forward is an open research problem usually cracked by methods this app does not have yet (off-policy continuous MARL like MADDPG/MATD3, or PPO with a centralized critic, i.e. MAPPO). So do not be surprised if, even after a long run, the walkers settle into a stable non-walking pose and the AI Skill meter (which measures real forward progress) stays low — that is the state of the art for plain parameter-sharing PPO here, not a bug.',
        cz: 'Jen PPO — Multiwalker je *kooperativní* roj se spojitým řízením, takže (jako Pursuit) PPO trénuje jeden sdílený „mozek“, který řídí všechny tři chodce najednou (sdílení parametrů). Neuroevoluce ani Q-učení nemají více­agentní cestu, takže jsou vypnuté. Poznámka k obtížnosti vzhledem k algoritmu: tahle úloha je pro PPO opravdu těžká. Spolehlivě zvládne tu *snazší* půlku — přestat padat — ale naučit se balík skutečně odnést dopředu je otevřený výzkumný problém, který obvykle řeší metody, jež tahle appka zatím nemá (off-policy spojité MARL jako MADDPG/MATD3, nebo PPO s centralizovaným kritikem, tj. MAPPO). Takže se nedivte, když se i po dlouhém běhu chodci ustálí ve stabilní nechodící pozici a měřič „AI Skill“ (který měří skutečný posun vpřed) zůstane nízko — to je u prostého parameter-sharing PPO současný strop, ne chyba.',
      },
      waterworld: {
        en: 'PPO only — Waterworld is a *cooperative* swarm with continuous control, so (like Pursuit and Multiwalker) PPO trains one shared "brain" that drives all five swimmers at once (parameter sharing). Neuroevolution and Q-learning have no multi-agent path, so they are turned off. The cooperative twist is the whole challenge: it takes two swimmers touching a food blob at the same moment to eat it, so the policy has to learn to pair up — a credit-assignment problem (which swimmer earned the reward?) that is genuinely hard for plain parameter-sharing PPO. Expect it to clearly beat random — dodging poison and making some coordinated catches — but tight cooperative foraging is the kind of task usually cracked by methods this app does not have yet (off-policy continuous MARL or a centralized critic), so the AI Skill meter may stay modest even after a long run.',
        cz: 'Jen PPO — Waterworld je *kooperativní* roj se spojitým řízením, takže (jako Pursuit a Multiwalker) PPO trénuje jeden sdílený „mozek“, který řídí všech pět plavců najednou (sdílení parametrů). Neuroevoluce ani Q-učení nemají více­agentní cestu, takže jsou vypnuté. Kooperativní háček je celá výzva: chuchvalce jídla se musí dotknout dva plavci ve stejný okamžik, aby ho snědli, takže se strategie musí naučit párovat — a to je problém přiřazení zásluh (který plavec si odměnu zasloužil?), který je pro prosté parameter-sharing PPO opravdu těžký. Čekejte, že jasně překoná náhodu — vyhýbá se jedu a udělá pár koordinovaných úlovků — ale těsný kooperativní lov je úloha, kterou obvykle řeší metody, jež tahle appka zatím nemá (off-policy spojité MARL nebo centralizovaný kritik), takže měřič „AI Skill“ může i po dlouhém běhu zůstat skromný.',
      },
      hopper: MUJOCO_ALGO,
      walker2d: MUJOCO_ALGO,
      halfcheetah: MUJOCO_ALGO,
      ant: MUJOCO_ALGO,
      reacher: MUJOCO_ALGO,
      swimmer: MUJOCO_ALGO,
      humanoid: MUJOCO_ALGO,
      tictactoe: {
        en: 'Both learn Tic-Tac-Toe. **PPO** learns by playing the built-in search AI (its teacher) and reaches near-perfect, always-drawing play. **AlphaZero** learns purely by **playing itself**, with look-ahead search guiding every move — on this tiny game both reach the draw ceiling, but AlphaZero is the more powerful method and a fun side-by-side comparison. (AlphaZero now trains on the GPU, playing many self-play games in parallel.)',
        cz: 'Piškvorky se naučí obě. **PPO** se učí hrou proti vestavěné prohledávací AI (svému učiteli) a dosáhne téměř dokonalé, vždy remízující hry. **AlphaZero** se učí čistě **hrou sám proti sobě**, kde každý tah řídí prohledávání dopředu — u téhle drobné hry dosáhnou obě remízového stropu, ale AlphaZero je silnější metoda a pěkné srovnání vedle sebe. (AlphaZero teď trénuje na GPU a hraje mnoho self-play partií najednou.)',
      },
      connect_four: {
        en: 'A great place to compare the two. **PPO** learns by playing the built-in search AI and climbs to beat the easy bot. **AlphaZero** learns by **playing itself** and, because it *searches ahead* even while playing (not just reacting), it usually ends up the stronger player — watch its curve climb past PPO\'s on the same scoreboard. (AlphaZero now trains on the GPU, playing many self-play games in parallel.)',
        cz: 'Skvělé místo pro porovnání obou. **PPO** se učí hrou proti vestavěné prohledávací AI a vyšplhá na úroveň, kde poráží lehkého bota. **AlphaZero** se učí **hrou sám proti sobě**, a protože *prohledává dopředu* i při hře (nejen reaguje), bývá nakonec silnějším hráčem — sledujte, jak jeho křivka přeroste PPO na stejné výsledkové tabuli. (AlphaZero teď trénuje na GPU a hraje mnoho self-play partií najednou.)',
      },
      othello: {
        en: 'PPO only here — it learns by playing the built-in search AI. **AlphaZero** is offered on the *smaller* boards (Tic-Tac-Toe, Connect Four), where it learns well and clearly beats PPO; on this huge 8×8 game its self-play needs far more games than a comfortable run allows before it shows real progress, so it is held back for a stronger future build.',
        cz: 'Tady jen PPO — učí se hrou proti vestavěné prohledávací AI. **AlphaZero** je k dispozici na *menších* deskách (Piškvorky, Čtyři v řadě), kde se učí dobře a jasně poráží PPO; u téhle obrovské hry 8×8 potřebuje jeho self-play mnohem víc partií, než pohodlný běh dovolí, aby ukázal skutečný pokrok, takže si ho šetříme pro silnější budoucí verzi.',
      },
      breakthrough: {
        en: 'PPO only here — it learns by playing the built-in search AI. **AlphaZero** is offered on the *smaller* boards (Tic-Tac-Toe, Connect Four), where it shines. Breakthrough\'s board has a huge number of possible moves, which makes AlphaZero\'s learn-from-yourself signal too noisy at a comfortable run length (it can even end up worse than an untrained net), so it waits for a stronger, longer AlphaZero setup.',
        cz: 'Tady jen PPO — učí se hrou proti vestavěné prohledávací AI. **AlphaZero** je k dispozici na *menších* deskách (Piškvorky, Čtyři v řadě), kde vyniká. Breakthrough má obrovské množství možných tahů, což dělá signál „uč se sám ze sebe" u AlphaZero při pohodlné délce běhu příliš zašuměný (může skončit i hůř než nenatrénovaná síť), takže čeká na silnější a delší nastavení AlphaZero.',
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
      carracing: {
        en: 'The default 3e-4 is a fine starting point for CarRacing\'s CnnPolicy too — keep it moderate, as learning from pixels is sensitive and too large a step can destabilise it. (Training runs on a GPU.)',
        cz: 'Výchozí 3e-4 je dobrý výchozí bod i pro CnnPolicy u CarRacing — držte ji umírněnou, protože učení z pixelů je citlivé a příliš velký krok ho může rozhodit. (Trénink běží na GPU.)',
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
      hopper: MUJOCO_LR,
      walker2d: MUJOCO_LR,
      halfcheetah: MUJOCO_LR,
      ant: MUJOCO_LR,
      reacher: MUJOCO_LR,
      swimmer: MUJOCO_LR,
      humanoid: MUJOCO_LR,
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
      carracing: {
        en: 'Keep γ high (0.99): a fast lap is a long sequence of turns, so the agent must value the track tiles many steps ahead over the small per-frame cost it pays now.',
        cz: 'Nechte γ vysoké (0.99): rychlé kolo je dlouhá řada zatáček, takže agent musí cenit dílky trati mnoho kroků dopředu víc než malou cenu za každý snímek teď.',
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
      hopper: MUJOCO_GAMMA,
      walker2d: MUJOCO_GAMMA,
      halfcheetah: MUJOCO_GAMMA,
      ant: MUJOCO_GAMMA,
      reacher: MUJOCO_GAMMA,
      swimmer: MUJOCO_GAMMA,
      humanoid: MUJOCO_GAMMA,
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
      carracing: {
        en: 'Leave it at 0.2 — learning from pixels can be unstable, and the clip is what keeps one bad batch from wrecking a working driving policy; tune budget and learning rate first.',
        cz: 'Nechte ji na 0.2 — učení z pixelů může být nestabilní a ořezávání brání jedné špatné dávce zničit funkční strategii řízení; laďte nejdřív rozpočet a rychlost učení.',
      },
      hopper: MUJOCO_CLIP,
      walker2d: MUJOCO_CLIP,
      halfcheetah: MUJOCO_CLIP,
      ant: MUJOCO_CLIP,
      reacher: MUJOCO_CLIP,
      swimmer: MUJOCO_CLIP,
      humanoid: MUJOCO_CLIP,
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
      carracing: {
        en: 'PPO\'s continuous action head already explores with its own noise, so 0 can work; a small bonus (e.g. 0.01) can help it try the gas early instead of crawling. Keep it modest — too much randomness just sends the car off the track.',
        cz: 'Spojitá akční hlava PPO už zkoumá vlastním šumem, takže 0 může stačit; drobný bonus (např. 0,01) může pomoci zkusit plyn dřív než se plížit. Držte ho mírný — moc náhodnosti auto jen vyveze z trati.',
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
      hopper: MUJOCO_ENT,
      walker2d: MUJOCO_ENT,
      halfcheetah: MUJOCO_ENT,
      ant: MUJOCO_ENT,
      reacher: MUJOCO_ENT,
      swimmer: MUJOCO_ENT,
      humanoid: MUJOCO_ENT,
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

  sweep_count: {
    general: {
      en: '**Seed sweep** — run this *same* configuration several times, changing only the random '
        + 'seed each time (the seed above, then +1, +2, …).\n'
        + 'A single run can get lucky or unlucky, so one curve never tells you how good a setup *really* '
        + 'is. Training the same setup across a few seeds shows the true picture: how high it usually '
        + 'gets, and how much run-to-run variance there is.\n'
        + 'The runs are **queued** and trained one after another (never in parallel — that would share '
        + 'the GPU/CPU and slow everything down), all tagged as one **experiment** so they can be '
        + 'aggregated and compared later. Press **Cancel** to stop the current seed and drop the rest.',
      cz: '**Série semen** — spustí *stejnou* konfiguraci několikrát a mění jen náhodné semeno (semeno '
        + 'výše, pak +1, +2, …).\n'
        + 'Jeden běh může mít štěstí nebo smůlu, takže jedna křivka nikdy neřekne, jak je nastavení '
        + '*doopravdy* dobré. Trénink stejného nastavení přes několik semen ukáže skutečný obrázek: kam '
        + 'se to obvykle dostane a jak velký je rozptyl mezi běhy.\n'
        + 'Běhy se **zařadí do fronty** a trénují jeden po druhém (nikdy paralelně — to by sdílelo '
        + 'GPU/CPU a vše zpomalilo), všechny označené jako jeden **experiment**, aby se daly později '
        + 'souhrnně porovnat. Tlačítkem **Zrušit** zastavíte aktuální semeno a zbytek zahodíte.',
    },
    recommended: {
      en: '3 seeds for a quick thesis-grade check; 5–10 for a publication-grade comparison.',
      cz: '3 semena pro rychlé ověření na úrovni bakalářky; 5–10 pro porovnání na úrovni publikace.',
    },
    range: '1 – 20',
  },

  total_steps: {
    general: {
      en: '**Training budget** — how many environment steps the agent gets to learn from before stopping.\nMore steps mean more practice and usually a better policy, up to the point where it has mastered the task.\n**SAC** and **TD3** are off-policy and far more sample-efficient than PPO, so their recommended budget is much smaller (they reuse past experience from a replay buffer) — the ★ here already reflects that when either is selected.',
      cz: '**Tréninkový rozpočet** — kolik kroků v prostředí agent dostane na učení, než se zastaví.\nVíce kroků znamená více cviku a obvykle lepší strategii, dokud úlohu nezvládne.\n**SAC** a **TD3** jsou off-policy a mnohem úspornější na data než PPO, takže jejich doporučený rozpočet je výrazně menší (znovu využívají minulé zkušenosti z paměti) — ★ to při zvolení kterékoli z nich už zohledňuje.',
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
      carracing: {
        en: 'CarRacing learns from pixels, so it needs a lot of practice — a solid driving policy typically takes around a million steps, hence the large ★ budget. Training runs on a GPU (Run needs a CUDA machine).',
        cz: 'CarRacing se učí z pixelů, takže potřebuje hodně cviku — slušná strategie řízení obvykle zabere kolem milionu kroků, odtud velký ★ rozpočet. Trénink běží na GPU (Spustit vyžaduje stroj s CUDA).',
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
      mpe_spread: {
        en: 'Cooperative coverage is slow to master — the ★ 500k gives a visible improvement on this CPU, but genuinely tidy coverage needs more (1–2M), which is what the GPU desktop is for. Here the step counter measures agent-steps summed across all agents, so it climbs fast; watch the reward trend rather than the raw count.',
        cz: 'Kooperativní pokrytí se zvládá pomalu — ★ 500k dá na tomto CPU viditelné zlepšení, ale opravdu úhledné pokrytí chce víc (1–2M), na což je určen stroj s GPU. Počítadlo kroků tu měří kroky agentů sečtené přes všechny agenty, takže roste rychle; sledujte spíš trend odměny než holé číslo.',
      },
      mpe_spread_swarm: {
        en: 'Six agents and six targets is much harder to coordinate, so it needs the larger ★ 1M budget — and even then expect only partial coverage on a CPU. This row is really a showcase of the swarm at scale; the heavy training belongs on the GPU desktop (G7b).',
        cz: 'Šest agentů a šest cílů se koordinuje mnohem hůř, takže potřebuje větší rozpočet ★ 1M — a i tak čekejte na CPU jen částečné pokrytí. Tento řádek je hlavně ukázka roje ve větším měřítku; náročný trénink patří na stroj s GPU (G7b).',
      },
      mpe_tag: {
        en: 'This budget applies once the per-species trainer lands (next step) — predators and prey learn together, so it takes a fair number of steps before the chase looks purposeful. The ★ 500k is a sensible CPU start; the GPU desktop scales it up.',
        cz: 'Tento rozpočet platí, jakmile přijde trenér pro jednotlivé druhy (další krok) — predátoři a kořist se učí společně, takže než honička začne vypadat účelně, chvíli to trvá. ★ 500k je rozumný start na CPU; stroj s GPU ho navýší.',
      },
      mpe_tag_pack: {
        en: 'The bigger pack-vs-prey world needs the larger ★ 1M budget once training is built (next step), and is really a showcase to scale on the GPU desktop.',
        cz: 'Větší svět smečka vs. kořist potřebuje větší rozpočet ★ 1M, jakmile bude trénink hotový (další krok), a je hlavně ukázkou pro škálování na stroji s GPU.',
      },
      pursuit: {
        en: 'Early on the swarm just drifts; with training the pursuers learn to fan out and actively chase the evaders, and the reward climbs off the floor. Even a few hundred thousand steps shows visible hunting on this CPU; the ★ 1M budget gives it room to get genuinely good. The step counter sums steps across all eight pursuers, so it climbs fast; watch the reward trend rather than the raw count.',
        cz: 'Zpočátku roj jen bloudí; s tréninkem se pronásledovatelé naučí rozprostřít a aktivně honit kořist a odměna se odrazí ode dna. I pár set tisíc kroků ukáže na tomto CPU viditelný lov; rozpočet ★ 1M mu dá prostor být opravdu dobrý. Počítadlo kroků sčítá kroky přes všech osm pronásledovatelů, takže roste rychle; sledujte spíš trend odměny než holé číslo.',
      },
      multiwalker: {
        en: 'Continuous balance is slow to learn, so Multiwalker needs the big ★ 2M budget. Learning comes in two stages, and they are not the same: first the walkers stop tipping over (the reward chart climbs off its −100 floor), and only much later — if at all at this budget — do they learn to actually walk the package forward. The AI Skill meter measures that second part (forward progress), so it can sit near 0% even while the reward chart looks healthy, because "doesn\'t fall" is not yet "walks". The step counter sums steps across all three walkers, so it climbs fast; watch the reward trend and the package, not the raw count, and give it plenty of time — real walking is genuinely hard here.',
        cz: 'Spojitá rovnováha se učí pomalu, takže Multiwalker potřebuje velký rozpočet ★ 2M. Učení má dvě fáze a nejsou totéž: nejdřív chodci přestanou padat (graf odměny se odrazí ode dna −100) a teprve mnohem později — pokud vůbec na tomto rozpočtu — se naučí balík skutečně posunout dopředu. Měřič „AI Skill“ měří tu druhou část (posun vpřed), takže může zůstat blízko 0 %, i když graf odměny vypadá zdravě, protože „nespadne“ ještě není „jde“. Počítadlo kroků sčítá kroky přes všechny tři chodce, takže roste rychle; sledujte trend odměny a balík, ne holé číslo, a dejte tomu hodně času — skutečná chůze je tu opravdu těžká.',
      },
      waterworld: {
        en: 'Cooperative foraging is slow to learn, so Waterworld needs the big ★ 2M budget. Early on the swimmers drift and waste energy (random scores below even a do-nothing swimmer, which just floats); with training they learn to dodge poison and pair up on food, and the reward climbs off the floor. But getting two swimmers onto the same food blob at the right moment, over and over, is hard for plain parameter-sharing PPO, so the AI Skill meter (which measures real cooperative catching) may stay modest even as the reward trend improves. The step counter sums steps across all five swimmers, so it climbs fast; watch the reward trend, not the raw count, and give it plenty of time.',
        cz: 'Kooperativní lov se učí pomalu, takže Waterworld potřebuje velký rozpočet ★ 2M. Zpočátku plavci bloudí a plýtvají energií (náhoda skóruje hůř než plavec, který nic nedělá a jen se vznáší); s tréninkem se naučí vyhýbat jedu a párovat se na jídle a odměna se odrazí ode dna. Ale dostat dva plavce na stejný chuchvalec ve správný okamžik, znovu a znovu, je pro prosté parameter-sharing PPO těžké, takže měřič „AI Skill“ (který měří skutečné kooperativní lovení) může zůstat skromný, i když se trend odměny zlepšuje. Počítadlo kroků sčítá kroky přes všech pět plavců, takže roste rychle; sledujte trend odměny, ne holé číslo, a dejte tomu hodně času.',
      },
      hopper: MUJOCO_STEPS,
      walker2d: MUJOCO_STEPS,
      halfcheetah: MUJOCO_STEPS,
      ant: MUJOCO_STEPS,
      reacher: MUJOCO_STEPS,
      swimmer: MUJOCO_STEPS,
      humanoid: {
        en: 'Humanoid is the hardest task here — seventeen joints to coordinate — so it carries the '
          + 'largest ★ budget (about 5 million steps), and even that only gets PPO started; it may not '
          + 'fully master walking at a practical budget. Expect slow, gradual progress and be patient. '
          + '(Training runs on a GPU machine; here Run is disabled until then.)',
        cz: 'Humanoid je tady nejtěžší úloha — sedmnáct kloubů ke koordinaci — takže nese největší '
          + 'rozpočet ★ (asi 5 milionů kroků) a ani ten PPO jen rozjede; chůzi nemusí při praktickém '
          + 'rozpočtu plně zvládnout. Čekejte pomalý, postupný pokrok a buďte trpěliví. (Trénink běží '
          + 'na stroji s GPU; tady je Spustit do té doby zakázané.)',
      },
    },
  },

  // ── SAC settings (S5a — Soft Actor-Critic, continuous control) ────────────

  sac_tau: {
    general: {
      en: '**Target-network update rate (τ).** SAC keeps a slow-moving "target" copy of its value network for stable learning. τ is how much the target drifts toward the live network each step.\nToo high → the target chases the live net and training wobbles. Too low → the target lags and learning crawls. The tiny default is almost always right.',
      cz: '**Rychlost aktualizace cílové sítě (τ).** SAC si drží pomalu se měnící „cílovou“ kopii své hodnotové sítě pro stabilní učení. τ udává, jak moc se cíl každý krok posune k živé síti.\nPříliš vysoká → cíl honí živou síť a učení se rozkmitá. Příliš nízká → cíl zaostává a učení se vleče. Drobná výchozí hodnota je téměř vždy správně.',
    },
    recommended: {
      en: 'The ★ default 0.005 is the standard SAC value — keep it unless you have a specific reason to change it.',
      cz: 'Doporučená ★ výchozí hodnota 0,005 je standardní hodnota SAC — neměňte ji, pokud k tomu nemáte konkrétní důvod.',
    },
    range: '0.001 – 0.05',
  },

  sac_buffer_size: {
    general: {
      en: '**Replay buffer size.** SAC is *off-policy*: it stores past transitions (state, action, reward, next state) in a big buffer and re-learns from random samples of them, which is why it needs far fewer real steps than PPO.\nA bigger buffer remembers more varied experience (more stable) but uses more memory; a small one forgets old lessons quickly.',
      cz: '**Velikost paměti přehrávání (replay buffer).** SAC je *off-policy*: minulé přechody (stav, akce, odměna, další stav) ukládá do velké paměti a znovu se učí z jejich náhodných vzorků, a právě proto potřebuje mnohem méně reálných kroků než PPO.\nVětší paměť si pamatuje pestřejší zkušenost (stabilnější), ale zabere víc paměti; malá rychle zapomíná staré lekce.',
    },
    recommended: {
      en: 'The ★ default (1M transitions) is the standard for continuous control. Lower it only if memory is tight.',
      cz: 'Doporučená ★ výchozí hodnota (1M přechodů) je standard pro spojité řízení. Snižte ji jen při nedostatku paměti.',
    },
    range: '100k – 1M',
  },

  sac_train_freq: {
    general: {
      en: '**Update frequency.** How many environment steps SAC collects before it runs a learning update (it does one gradient step per collected step, so the ratio stays balanced).\n1 = learn after every step (the most sample-efficient, the default). Larger values collect more before each update — a little faster in wall-clock, a little less sample-efficient.',
      cz: '**Frekvence aktualizace.** Kolik kroků v prostředí SAC nasbírá, než spustí učící aktualizaci (na každý nasbíraný krok udělá jeden gradientní krok, takže poměr zůstává vyvážený).\n1 = učit se po každém kroku (nejúspornější na data, výchozí). Vyšší hodnoty nasbírají víc před každou aktualizací — o něco rychlejší v reálném čase, o něco méně úsporné na data.',
    },
    recommended: {
      en: 'The ★ default 1 is the most sample-efficient and the usual choice. Raise it only to trade a little efficiency for throughput.',
      cz: 'Doporučená ★ výchozí hodnota 1 je nejúspornější na data a obvyklá volba. Zvyšte ji jen pro výměnu trochy úspornosti za propustnost.',
    },
    range: '1 – 64',
  },

  sac_ent_coef: {
    general: {
      en: '**Entropy temperature.** SAC deliberately rewards the agent for staying a little unpredictable (high "entropy"), which keeps it exploring instead of locking onto one habit too early. This controls how strong that bonus is.\n**auto** lets SAC tune it automatically as training goes — this is SAC\'s signature trick and almost always the best choice. A fixed number pins it instead (higher = more random/exploratory, lower = more decisive).',
      cz: '**Teplota entropie.** SAC záměrně odměňuje agenta za to, že zůstane trochu nepředvídatelný (vysoká „entropie“), což ho nutí dál zkoumat místo brzkého zaseknutí na jednom zvyku. Tohle řídí, jak silný ten bonus je.\n**auto** nechá SAC hodnotu ladit automaticky během tréninku — to je typický trik SAC a téměř vždy nejlepší volba. Pevné číslo ji místo toho zafixuje (vyšší = náhodnější/zkoumavější, nižší = rozhodnější).',
    },
    recommended: {
      en: 'Leave it on **auto** (★) — self-tuned entropy is one of the main reasons SAC works so well. Pin a fixed value only to experiment.',
      cz: 'Nechte na **auto** (★) — samoladěná entropie je jedním z hlavních důvodů, proč SAC funguje tak dobře. Pevnou hodnotu zafixujte jen kvůli experimentu.',
    },
  },

  // ── TD3 settings (S5b — Twin Delayed DDPG, continuous control) ────────────

  td3_tau: {
    general: {
      en: '**Target-network update rate (τ).** Like SAC, TD3 keeps slow-moving "target" copies of its value networks for stable learning. τ is how much each target drifts toward the live network every step.\nToo high → the target chases the live net and training wobbles. Too low → the target lags and learning crawls. The tiny default is almost always right.',
      cz: '**Rychlost aktualizace cílové sítě (τ).** Stejně jako SAC si TD3 drží pomalu se měnící „cílové“ kopie svých hodnotových sítí pro stabilní učení. τ udává, jak moc se každý cíl každý krok posune k živé síti.\nPříliš vysoká → cíl honí živou síť a učení se rozkmitá. Příliš nízká → cíl zaostává a učení se vleče. Drobná výchozí hodnota je téměř vždy správně.',
    },
    recommended: {
      en: 'The ★ default 0.005 is the standard value — keep it unless you have a specific reason to change it.',
      cz: 'Doporučená ★ výchozí hodnota 0,005 je standardní hodnota — neměňte ji, pokud k tomu nemáte konkrétní důvod.',
    },
    range: '0.001 – 0.05',
  },

  td3_buffer_size: {
    general: {
      en: '**Replay buffer size.** TD3 is *off-policy*: it stores past transitions (state, action, reward, next state) in a big buffer and re-learns from random samples of them, which is why it needs far fewer real steps than PPO.\nA bigger buffer remembers more varied experience (more stable) but uses more memory; a small one forgets old lessons quickly.',
      cz: '**Velikost paměti přehrávání (replay buffer).** TD3 je *off-policy*: minulé přechody (stav, akce, odměna, další stav) ukládá do velké paměti a znovu se učí z jejich náhodných vzorků, a právě proto potřebuje mnohem méně reálných kroků než PPO.\nVětší paměť si pamatuje pestřejší zkušenost (stabilnější), ale zabere víc paměti; malá rychle zapomíná staré lekce.',
    },
    recommended: {
      en: 'The ★ default (1M transitions) is the standard for continuous control. Lower it only if memory is tight.',
      cz: 'Doporučená ★ výchozí hodnota (1M přechodů) je standard pro spojité řízení. Snižte ji jen při nedostatku paměti.',
    },
    range: '100k – 1M',
  },

  td3_train_freq: {
    general: {
      en: '**Update frequency.** How many environment steps TD3 collects before it runs a learning update (it does one gradient step per collected step, so the ratio stays balanced).\n1 = learn after every step (the most sample-efficient, the default). Larger values collect more before each update — a little faster in wall-clock, a little less sample-efficient.',
      cz: '**Frekvence aktualizace.** Kolik kroků v prostředí TD3 nasbírá, než spustí učící aktualizaci (na každý nasbíraný krok udělá jeden gradientní krok, takže poměr zůstává vyvážený).\n1 = učit se po každém kroku (nejúspornější na data, výchozí). Vyšší hodnoty nasbírají víc před každou aktualizací — o něco rychlejší v reálném čase, o něco méně úsporné na data.',
    },
    recommended: {
      en: 'The ★ default 1 is the most sample-efficient and the usual choice. Raise it only to trade a little efficiency for throughput.',
      cz: 'Doporučená ★ výchozí hodnota 1 je nejúspornější na data a obvyklá volba. Zvyšte ji jen pro výměnu trochy úspornosti za propustnost.',
    },
    range: '1 – 64',
  },

  td3_train_noise: {
    general: {
      en: '**Exploration noise.** TD3\'s policy is *deterministic* — for one state it always picks the same action — so it can\'t explore on its own. To discover new behaviour it adds a little random Gaussian noise to each action it takes while training. This sets how big that noise is.\nThis is TD3\'s counterpart to SAC\'s entropy bonus: a bit more keeps it exploring, too much makes its practice runs sloppy. 0 turns exploration noise off entirely (it then only explores during the random warm-up).',
      cz: '**Šum pro zkoumání.** Strategie TD3 je *deterministická* — pro jeden stav vždy zvolí stejnou akci — takže sama o sobě nezkoumá. Aby objevila nové chování, přidává během tréninku ke každé akci trochu náhodného gaussovského šumu. Tohle určuje, jak velký ten šum je.\nJe to obdoba bonusu za entropii u SAC: trochu víc ho udrží zkoumavým, příliš mnoho zaneřádí jeho tréninkové pokusy. 0 šum pro zkoumání úplně vypne (pak zkoumá jen během náhodného zahřívání).',
    },
    recommended: {
      en: 'The ★ default 0.1 is the standard TD3 / rl-zoo3 value and a good balance. Nudge it up if the agent gets stuck, down if its runs look too erratic.',
      cz: 'Doporučená ★ výchozí hodnota 0,1 je standardní hodnota TD3 / rl-zoo3 a dobrý kompromis. Lehce ji zvyšte, pokud agent uvázne, snižte, pokud jeho pokusy vypadají příliš nahodile.',
    },
    range: '0.0 – 0.5',
  },

  // ── DQN settings (S5c — Deep Q-Network, off-policy value-based, discrete actions) ──

  dqn_buffer_size: {
    general: {
      en: '**Replay buffer size.** DQN is *off-policy*: it stores past transitions (state, action, reward, next state) in a big buffer and re-learns from random samples of them, which is why it needs far fewer real steps than a fresh-data-only method.\nA bigger buffer remembers more varied experience (more stable) but uses more memory; a small one forgets old lessons quickly. It is kept smaller than SAC/TD3\'s default because picture-based games (Atari) store full frames here, which is memory-heavy.',
      cz: '**Velikost paměti přehrávání (replay buffer).** DQN je *off-policy*: minulé přechody (stav, akce, odměna, další stav) ukládá do velké paměti a znovu se učí z jejich náhodných vzorků, a právě proto potřebuje mnohem méně reálných kroků než metody učící se jen z čerstvých dat.\nVětší paměť si pamatuje pestřejší zkušenost (stabilnější), ale zabere víc paměti; malá rychle zapomíná staré lekce. Je menší než výchozí u SAC/TD3, protože hry z obrazu (Atari) tu ukládají celé snímky, což je náročné na paměť.',
    },
    recommended: {
      en: 'The ★ default (100k transitions) suits the small games. For Atari it stays modest on purpose — a full 1M buffer of stacked frames would need tens of GB of RAM.',
      cz: 'Doporučená ★ výchozí hodnota (100k přechodů) sedí malým hrám. U Atari zůstává záměrně skromná — plný 1M buffer skládaných snímků by potřeboval desítky GB RAM.',
    },
    range: '10k – 1M',
  },

  dqn_train_freq: {
    general: {
      en: '**Update frequency.** How many environment steps DQN collects before it runs a learning update. Smaller = learn more often (more sample-efficient, slower in wall-clock); larger = collect more between updates.\nUnlike the policy-gradient methods, DQN can do many gradient steps per collected step, so this trades how fresh each update\'s data is against throughput.',
      cz: '**Frekvence aktualizace.** Kolik kroků v prostředí DQN nasbírá, než spustí učící aktualizaci. Menší = učit se častěji (úspornější na data, pomalejší v reálném čase); větší = nasbírat víc mezi aktualizacemi.\nNa rozdíl od metod policy-gradient může DQN udělat na jeden nasbíraný krok více gradientních kroků, takže tohle balancuje čerstvost dat každé aktualizace proti propustnosti.',
    },
    recommended: {
      en: 'Per-game ★ from rl-zoo3\'s tuned recipes — CartPole likes a large value (256), while Acrobot/LunarLander/Atari use 4. Keep the recommendation unless you want to experiment.',
      cz: 'Pro každou hru ★ podle laděných receptů z rl-zoo3 — CartPole má rád velkou hodnotu (256), kdežto Acrobot/LunarLander/Atari používají 4. Doporučení neměňte, pokud nechcete experimentovat.',
    },
    range: '1 – 256',
  },

  dqn_target_update: {
    general: {
      en: '**Target-network sync.** DQN keeps a slow "target" copy of its value network so the goal it trains toward does not move every step (which would make learning unstable). This is how often (in steps) that target is hard-copied from the live network.\nToo frequent → the target chases the live net and training wobbles. Too rare → the target is stale and learning lags. (This is DQN\'s blunt counterpart to SAC/TD3\'s smooth τ update.)',
      cz: '**Synchronizace cílové sítě.** DQN si drží pomalou „cílovou“ kopii své hodnotové sítě, aby se cíl, ke kterému trénuje, neměnil každý krok (což by učení rozkolísalo). Tohle udává, jak často (v krocích) se cíl natvrdo zkopíruje z živé sítě.\nPříliš často → cíl honí živou síť a učení se rozkmitá. Příliš zřídka → cíl je zastaralý a učení zaostává. (Je to hrubší obdoba plynulé aktualizace τ u SAC/TD3.)',
    },
    recommended: {
      en: 'Per-game ★ from rl-zoo3 — CartPole syncs very often (10), the other classics around 250–600, Atari rarely (10000). Keep the recommendation.',
      cz: 'Pro každou hru ★ podle rl-zoo3 — CartPole synchronizuje velmi často (10), ostatní klasiky kolem 250–600, Atari zřídka (10000). Doporučení ponechte.',
    },
    range: '1 – 20000 (per game)',
  },

  dqn_exploration_fraction: {
    general: {
      en: '**Exploration schedule (ε-greedy).** DQN explores by sometimes acting at random: it plays a random action with probability ε. ε starts at 1.0 (all random) and falls to the final value below; this sets **over what fraction of the whole training run** that fall happens. After that, ε holds at the final value.\nA longer fraction = more exploration before settling down. This is DQN\'s distinctive exploration knob — its counterpart to SAC\'s entropy and TD3\'s action noise.',
      cz: '**Plán zkoumání (ε-greedy).** DQN zkoumá tím, že občas jedná náhodně: s pravděpodobností ε zahraje náhodnou akci. ε začíná na 1,0 (vše náhodně) a klesá ke koncové hodnotě níže; tohle určuje, **přes jakou část celého tréninku** ten pokles proběhne. Poté ε drží na koncové hodnotě.\nDelší podíl = víc zkoumání, než se to ustálí. To je pro DQN typický nástroj zkoumání — obdoba entropie u SAC a šumu akcí u TD3.',
    },
    recommended: {
      en: 'Per-game ★ from rl-zoo3 (around 0.1–0.2). Raise it if the agent seems to settle on a poor habit too early; lower it if it stays random too long.',
      cz: 'Pro každou hru ★ podle rl-zoo3 (kolem 0,1–0,2). Zvyšte, pokud se agent příliš brzy zasekne na špatném zvyku; snižte, pokud zůstává náhodný příliš dlouho.',
    },
    range: '0.01 – 0.5',
  },

  dqn_exploration_final_eps: {
    general: {
      en: '**Final exploration rate.** The value ε settles at once the schedule above finishes — the residual chance the agent keeps acting at random for the rest of training, so it never stops exploring entirely.\nHigher = more lingering randomness (safer against getting stuck, but noisier play); lower = more decisive, greedier behaviour late in training.',
      cz: '**Koncová míra zkoumání.** Hodnota, na které se ε ustálí po dokončení plánu výše — zbytková šance, že agent po zbytek tréninku stále občas jedná náhodně, takže nikdy úplně nepřestane zkoumat.\nVyšší = víc přetrvávající náhodnosti (bezpečnější proti zaseknutí, ale rozkolísanější hra); nižší = rozhodnější, hladovější chování v pozdní fázi tréninku.',
    },
    recommended: {
      en: 'Per-game ★ from rl-zoo3 (around 0.01–0.1). The small Atari value (0.01) makes late play nearly greedy; the classics keep a touch more.',
      cz: 'Pro každou hru ★ podle rl-zoo3 (kolem 0,01–0,1). Malá hodnota u Atari (0,01) dělá pozdní hru téměř hladovou; klasiky si nechávají o trošku víc.',
    },
    range: '0.0 – 0.2',
  },

  a2c_n_steps: {
    general: {
      en: '**Rollout length.** How many steps A2C plays before each learning update. This is A2C\'s signature knob: it defaults to just a handful of steps (5), far shorter than PPO\'s big 2048-step rollout — A2C updates little and often.\nShort rollouts make each update cheap but noisy (few samples to estimate the gradient from); longer rollouts steady the gradient at the cost of updating less often. The classic A2C recipe leans on many parallel environments to compensate; here we run a single environment, so the ★ value is nudged up a little to keep learning stable.',
      cz: '**Délka rolloutu.** Kolik kroků A2C odehraje před každou učící aktualizací. To je pro A2C typický nástroj: výchozí je jen pár kroků (5), mnohem méně než velký 2048krokový rollout u PPO — A2C aktualizuje po malých dávkách a často.\nKrátké rollouty dělají každou aktualizaci levnou, ale rozkolísanou (málo vzorků na odhad gradientu); delší rollouty gradient ustálí za cenu řidších aktualizací. Klasický recept A2C to kompenzuje mnoha paralelními prostředími; my běžíme jedno, takže je ★ hodnota trochu zvýšená, aby učení zůstalo stabilní.',
    },
    recommended: {
      en: 'Per-game ★ (nudged up from the bare 5 for the single-environment setup — e.g. 32 on CartPole). Lower it toward 5 for the textbook high-variance A2C; raise it to steady a noisy run.',
      cz: 'Pro každou hru ★ (zvýšená z holé 5 kvůli běhu s jedním prostředím — např. 32 u CartPole). Snižte k 5 pro učebnicové rozkolísané A2C; zvyšte, když je běh moc rozkmitaný.',
    },
    range: '5 – 128',
  },

  a2c_gae_lambda: {
    general: {
      en: '**Return smoothing (GAE λ).** Controls how A2C estimates how good each move was. At λ = 1.0 it uses the *full* actual outcome of the episode (classic A2C — unbiased but high-variance); lower values blend in the value network\'s own predictions (Generalized Advantage Estimation), trading a little bias for much less variance — the same mechanism PPO uses at 0.95.\nHigher = truer but noisier signal; lower = smoother but slightly biased. This is one of the few dials where you can make A2C behave more like PPO.',
      cz: '**Vyhlazení návratů (GAE λ).** Řídí, jak A2C odhaduje, jak dobrý každý tah byl. Při λ = 1,0 používá *celý* skutečný výsledek epizody (klasické A2C — nezkreslené, ale hodně rozkolísané); nižší hodnoty přimíchají vlastní předpovědi hodnotové sítě (Generalized Advantage Estimation) a vymění trochu zkreslení za mnohem menší rozptyl — stejný mechanismus, jaký PPO používá při 0,95.\nVyšší = věrnější, ale rozkolísanější signál; nižší = hladší, ale mírně zkreslený. Jeden z mála knoflíků, kterým A2C přiblížíte chování PPO.',
    },
    recommended: {
      en: 'The ★ default 1.0 is classic A2C (full Monte-Carlo returns). Lower it toward 0.95 if training is too noisy — that is exactly what PPO does.',
      cz: 'Doporučená ★ hodnota 1,0 je klasické A2C (plné Monte-Carlo návraty). Snižte k 0,95, pokud je trénink moc rozkolísaný — přesně to dělá PPO.',
    },
    range: '0.8 – 1.0',
  },

  // ── QR-DQN settings (S5e — Quantile-Regression DQN, distributional value-based) ──
  qrdqn_n_quantiles: {
    general: {
      en: '**Number of quantiles.** This is the one knob QR-DQN adds to DQN. Plain DQN estimates a single number per action — the *average* return it expects. QR-DQN instead estimates the whole **spread** of possible returns, described by this many evenly-spaced points (quantiles): the median, the quartiles, and so on. It still *chooses* the action with the highest average, but learning the full distribution is often a richer, more stable training signal (it is one of the ingredients of the famous "Rainbow" agent).\nMore quantiles = a finer picture of the distribution (potentially better, but a slightly bigger, slower network); fewer = a coarser picture that trains faster. With just one quantile it would collapse back toward ordinary DQN.',
      cz: '**Počet kvantilů.** To je jediný knoflík, který QR-DQN přidává k DQN. Obyčejné DQN odhaduje jediné číslo na akci — *průměrný* očekávaný návrat. QR-DQN místo toho odhaduje celé **rozložení** možných návratů, popsané tímto počtem rovnoměrně rozmístěných bodů (kvantilů): medián, kvartily a tak dál. Akci pořád *vybírá* podle nejvyššího průměru, ale učení celé distribuce bývá bohatší a stabilnější učící signál (je to jedna ze složek slavného agenta „Rainbow“).\nVíce kvantilů = jemnější obraz distribuce (potenciálně lepší, ale o něco větší, pomalejší síť); méně = hrubší obraz, který se učí rychleji. S jediným kvantilem by se to vrátilo zpět k obyčejnému DQN.',
    },
    recommended: {
      en: 'The ★ classic-control default is 25 (the rl-zoo3 CartPole recipe uses 10); Atari uses 200 (the QR-DQN paper\'s value). More is rarely worth the extra cost on the small games.',
      cz: 'Doporučená ★ hodnota pro klasické řízení je 25 (recept rl-zoo3 pro CartPole používá 10); Atari používá 200 (hodnota z článku o QR-DQN). Víc se u malých her málokdy vyplatí.',
    },
    range: '5 – 200',
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

  rounds: {
    general: {
      en: '**Self-play rounds** — how many times the two species take turns learning.\nPredator and prey can\'t learn at the same time here, so training alternates: in each round the predators improve against a frozen copy of the current prey, then the prey improve against a frozen copy of the now-better predators. More rounds = a deeper back-and-forth arms race, but a longer run (the total step budget is split evenly across all the turns).',
      cz: '**Kola self-play** — kolikrát se oba druhy vystřídají v učení.\nPredátor a kořist se tu nemůžou učit zároveň, takže se trénink střídá: v každém kole se predátoři zlepšují proti zmrazené kopii současné kořisti, pak se kořist zlepšuje proti zmrazené kopii teď už lepších predátorů. Víc kol = hlubší vzájemné „závody ve zbrojení“, ale delší běh (celkový rozpočet kroků se rozdělí rovnoměrně mezi všechny tahy).',
    },
    recommended: {
      en: '8 is a good default — enough alternations to see both species visibly co-evolve. Raise it for a longer, deeper arms race; lower it for a quick look.',
      cz: '8 je dobrá výchozí hodnota — dost střídání, aby bylo vidět, jak se oba druhy společně vyvíjejí. Zvyšte pro delší, hlubší závod; snižte pro rychlý náhled.',
    },
    range: '2 – 20',
    perEnv: {
      mpe_tag: {
        en: 'Watch the two reward curves on the chart: the predators (red) should climb as they learn to corner the prey, while the prey (blue) tries to claw its score back up by escaping. Each round flips which curve is moving.',
        cz: 'Sledujte dvě křivky odměny v grafu: predátoři (červená) by měli stoupat, jak se učí kořist zahnat do kouta, zatímco kořist (modrá) se snaží svou skóre vyšplhat zpět únikem. Každé kolo přepne, která křivka se hýbe.',
      },
      mpe_tag_pack: {
        en: 'With a six-predator pack and two prey the coordination is richer, so a few more rounds (10–14) can help both species settle into stable hunting and evasion patterns.',
        cz: 'Se šestičlennou smečkou predátorů a dvěma kořistmi je koordinace bohatší, takže pár kol navíc (10–14) může oběma druhům pomoci ustálit se ve stabilních vzorcích lovu a úniku.',
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

  // ── AlphaZero-lite settings (G6f) ─────────────────────────────────────────
  // The 4th algorithm, on the board games: a CNN that learns purely from self-play, using
  // Monte-Carlo Tree Search guided by the net itself. These pair with the eval-vs-AI reward curve.

  iterations: {
    general: {
      en: '**Training budget** — how many rounds of "play a batch of games against yourself, then learn from them" AlphaZero does.\nEach iteration the AI plays a fresh batch of self-play games, then trains its network on what happened. More iterations = a stronger net, up to the point the small game is mastered. This is AlphaZero\'s equivalent of PPO\'s Total Steps (the full budget is iterations × games-per-iteration self-play games).',
      cz: '**Tréninkový rozpočet** — kolik kol „odehraj dávku partií sám proti sobě a pak se z nich uč“ AlphaZero udělá.\nV každé iteraci AI odehraje novou dávku self-play partií a pak na nich natrénuje svou síť. Víc iterací = silnější síť, dokud malou hru nezvládne. Je to obdoba PPO „Celkem kroků“ pro AlphaZero (celý rozpočet je iterace × her na iteraci self-play partií).',
    },
    recommended: {
      en: 'The ★ default is tuned per game. Watch the reward curve: while it is still climbing, more iterations keep making the net stronger; once it flattens near the top, it has learned what it can at this search depth. For a deep game like chess you can raise this a long way (up to 500) to train for hours — or Save and Load to continue a run later, picking up from the trained net.',
      cz: 'Výchozí ★ je laděná pro každou hru. Sledujte křivku odměny: dokud stoupá, další iterace síť dál posilují; jakmile se ustálí poblíž stropu, naučila se, co při této hloubce hledání zvládne. U hluboké hry jako šachy můžete jít hodně vysoko (až 500) a trénovat hodiny — nebo dát Uložit a Načíst a pokračovat v běhu později, navázat na natrénovanou síť.',
    },
    range: '5 – 500',
    perEnv: {
      tictactoe: {
        en: 'Tic-Tac-Toe is tiny, so the net reaches strong (drawing) play quickly — but it is scored against a tough reference AI, so it takes a few dozen iterations for the curve to climb from losses up toward 0 (draws), which is mastery here.',
        cz: 'Piškvorky jsou drobné, takže síť rychle dosáhne silné (remízové) hry — měří se ale proti tvrdší referenční AI, takže pár desítek iterací trvá, než křivka vystoupá z proher k 0 (remízy), což je tady mistrovství.',
      },
      connect_four: {
        en: 'Connect Four is bigger, so give it the full default (around 30) to watch the net climb from losing to clearly beating the reference AI. Because AlphaZero searches ahead even when playing, it usually overtakes the plain-PPO net on the same game.',
        cz: 'Čtyři v řadě je větší, takže jí dejte plný výchozí rozpočet (kolem 30) a sledujte, jak síť stoupá od proher k jasnému vítězství nad referenční AI. Protože AlphaZero hledá dopředu i při hře, obvykle předčí síť z prostého PPO na téže hře.',
      },
      chess: {
        en: 'Chess is enormous, so a short run only scratches the surface — the default is modest so a run finishes in a reasonable time and you can see the curve move. Each iteration here is real work (chess self-play games are long), so raising this makes a noticeably stronger but slower-to-train opponent. Reaching genuinely strong play takes a lot of self-play (think many iterations / a long run), so treat a short run as "plays legally and is learning", not "mastered".',
        cz: 'Šachy jsou obrovské, takže krátký běh jen načne povrch — výchozí hodnota je skromná, aby běh skončil v rozumném čase a vy viděli, jak se křivka hýbe. Každá iterace je tu skutečná práce (šachové self-play partie jsou dlouhé), takže zvýšení dá znatelně silnějšího, ale pomaleji trénovaného soupeře. Opravdu silná hra vyžaduje hodně self-play (spousta iterací / dlouhý běh), takže krátký běh berte jako „hraje legálně a učí se“, ne „zvládnuto“.',
      },
    },
  },

  gumbel_sims: {
    general: {
      en: '**Search depth** — how many moves AlphaZero "thinks ahead" before each of its own moves during self-play.\nAlphaZero pairs its network with a tree search: for every move it tries out this many continuations, lets the network judge them, and plays the one that looked best. More search = sharper moves to learn from, but slower self-play. This version uses **Gumbel search**, a smarter way of spending those simulations that finds a strong move with far fewer of them — so this dial sits much lower than a classic search would need.',
      cz: '**Hloubka hledání** — kolik tahů AlphaZero „promýšlí dopředu“ před každým svým tahem během self-play.\nAlphaZero spojuje svou síť se stromovým prohledáváním: u každého tahu vyzkouší tolik pokračování, nechá je síť ohodnotit a zahraje to, které vypadalo nejlépe. Víc hledání = ostřejší tahy k učení, ale pomalejší self-play. Tato verze používá **Gumbel hledání**, chytřejší způsob, jak ty simulace utratit, který najde silný tah s mnohem menším počtem — proto je tento volič mnohem níž, než by klasické hledání potřebovalo.',
    },
    recommended: {
      en: '16 is plenty thanks to Gumbel search — it reaches the move quality a plain search would need 50+ simulations for, at a fraction of the cost (so self-play runs roughly twice as fast). Raise it for an even deeper look on hard games; lower it for the quickest, lightest training.',
      cz: '16 bohatě stačí díky Gumbel hledání — dosáhne kvality tahu, na kterou by prosté hledání potřebovalo 50+ simulací, za zlomek nákladů (takže self-play běží zhruba dvakrát rychleji). Zvyšte pro ještě hlubší pohled u těžkých her; snižte pro nejrychlejší, nejlehčí trénink.',
    },
    range: '4 – 64',
    perEnv: {
      tictactoe: {
        en: 'Tic-Tac-Toe is trivial to read, so even the lightest search finds the right move — the default keeps self-play fast.',
        cz: 'Piškvorky se čtou triviálně, takže i to nejlehčí hledání najde správný tah — výchozí hodnota drží self-play rychlé.',
      },
      connect_four: {
        en: 'Connect Four has real tactics (traps, threats), so a touch more search noticeably sharpens the moves the net learns from — but Gumbel keeps even the default surprisingly strong.',
        cz: 'Čtyři v řadě má skutečnou taktiku (pasti, hrozby), takže o trochu víc hledání znatelně zostří tahy, ze kterých se síť učí — ale Gumbel drží i výchozí hodnotu překvapivě silnou.',
      },
      chess: {
        en: 'Chess is deep, so more search means sharper move targets — and it is the most expensive dial here (every simulation runs the big network again over a 64-square board). Gumbel is what makes the low default workable: raise it for stronger play once you are ready for a longer run.',
        cz: 'Šachy jsou hluboké, takže víc hledání znamená ostřejší cílové tahy — a je to tu nejdražší volič (každá simulace znovu spustí velkou síť nad 64 poli). Gumbel je to, co dělá nízkou výchozí hodnotu použitelnou: zvyšte ji pro silnější hru, až budete připraveni na delší běh.',
      },
    },
  },

  gumbel_considered: {
    general: {
      en: '**Considered moves** — how many candidate moves the search seriously compares before each of its own moves.\nRather than spread its thinking thinly over every legal move, Gumbel search shortlists this many of the most promising ones and runs a little knockout tournament between them (repeatedly dropping the weaker half), so the simulations go where they matter most. On positions with fewer legal moves than this, it simply considers them all.',
      cz: '**Zvažované tahy** — kolik kandidátních tahů hledání před každým svým tahem doopravdy porovná.\nMísto aby své přemýšlení tence rozprostřelo přes každý možný tah, Gumbel hledání vybere tolik nejnadějnějších a uspořádá mezi nimi malý vyřazovací turnaj (opakovaně vyřadí slabší polovinu), takže simulace jdou tam, kde nejvíc záleží. V pozicích s méně možnými tahy, než je tato hodnota, zváží prostě všechny.',
    },
    recommended: {
      en: '16 is a good balance: broad enough not to overlook a strong move, focused enough that each candidate gets a real look. On small boards it is automatically capped at the number of legal moves. Lower it to look harder at fewer moves; raise it on move-rich games to widen the shortlist.',
      cz: '16 je dobrý kompromis: dost široký, aby nepřehlédl silný tah, a dost soustředěný, aby každý kandidát dostal skutečný pohled. Na malých deskách se automaticky omezí na počet možných tahů. Snižte pro tvrdší pohled na méně tahů; zvyšte u her bohatých na tahy, abyste rozšířili užší výběr.',
    },
    range: '2 – 32',
    perEnv: {
      chess: {
        en: 'Chess often has 30+ legal moves, so this is where the setting bites: 16 focuses the search on the most promising moves instead of thinning it across all of them. Widen it to scan more candidates (slower), narrow it to think harder about a few.',
        cz: 'Šachy mají často 30+ možných tahů, takže právě tady se nastavení projeví: 16 soustředí hledání na nejnadějnější tahy místo toho, aby ho ztenčilo přes všechny. Rozšiřte pro prohledání více kandidátů (pomalejší), zužte pro tvrdší přemýšlení nad několika málo.',
      },
    },
  },

  games_per_iter: {
    general: {
      en: '**Self-play games per iteration** — how many fresh games the AI plays against itself before each round of learning.\nThese games are the AI\'s only training data — it has no human examples, it learns purely from its own play. More games per iteration = steadier, less noisy learning, but each iteration takes longer.',
      cz: '**Self-play partií na iteraci** — kolik nových partií AI odehraje sama proti sobě před každým kolem učení.\nTyto partie jsou jediná tréninková data AI — nemá žádné lidské příklady, učí se čistě z vlastní hry. Víc partií na iteraci = stabilnější, méně zašuměné učení, ale každá iterace trvá déle.',
    },
    recommended: {
      en: 'This also sets the **GPU batch width**: the self-play games run concurrently and their network evaluations are batched into one pass, so a higher value keeps the GPU fuller and the throughput higher (up to ~64, where chess hits its sweet spot). 24 suits the small games; chess defaults to 64 for ~2× the games-per-second.',
      cz: 'Tohle zároveň určuje **šířku dávky pro GPU**: self-play partie běží souběžně a jejich vyhodnocení sítí se sloučí do jednoho průchodu, takže vyšší hodnota drží GPU plnější a propustnost vyšší (asi do 64, kde mají šachy optimum). 24 sedí malým hrám; šachy mají výchozích 64 pro ~2× partií za sekundu.',
    },
    range: '8 – 48',
  },

  actor_processes: {
    general: {
      en: '**Self-play workers** — how many separate worker processes generate self-play games in parallel.\nGenerating the games is the slow part of training, and a single worker leaves your graphics card (GPU) only about half busy — it spends a lot of time waiting on one CPU core that drives the game logic. Running two workers side by side, each with its own copy of the network, lets the GPU stay busy while one worker thinks and the other computes, so games are produced faster. Needs a GPU; on a CPU-only machine this setting is ignored (one worker is used).',
      cz: '**Pracovní procesy self-play** — kolik samostatných procesů (worker processes) souběžně generuje self-play partie.\nGenerování partií je pomalá část tréninku a jeden worker nechá grafickou kartu (GPU) vytíženou jen asi z poloviny — hodně času čeká na jedno jádro procesoru (CPU core), které řídí logiku hry. Když běží dva workeři vedle sebe, každý s vlastní kopií sítě, GPU zůstává vytížené, zatímco jeden worker přemýšlí a druhý počítá, takže partie vznikají rychleji. Vyžaduje GPU; na počítači jen s CPU se nastavení ignoruje (použije se jeden worker).',
    },
    recommended: {
      en: '1 keeps the classic single-worker training (fine on every machine). On a GPU, **2 is the sweet spot** — it produces chess self-play roughly 1.6× faster and pushes GPU usage from ~50 % to ~95 %. Going higher does not help here (on Windows the GPU cannot be shared cleanly across more than two processes, so 3–4 actually run slower). Only the heavy games (chess) gain from this; the small boards already train fast with one worker.',
      cz: '1 zachová klasický trénink s jedním workerem (funguje na každém stroji). Na GPU je **2 optimum** — vyrobí šachové self-play zhruba 1,6× rychleji a zvedne vytížení GPU z ~50 % na ~95 %. Víc už tu nepomůže (ve Windows nejde GPU čistě sdílet mezi více než dvěma procesy, takže 3–4 běží naopak pomaleji). Těží z toho jen těžké hry (šachy); malé desky se s jedním workerem učí rychle už teď.',
    },
    range: '1 – 4',
    perEnv: {
      chess: {
        en: 'Chess is the one game heavy enough to benefit: its big network keeps two workers genuinely busy, so 2 is the recommended setting on a GPU for ~1.6× faster self-play.',
        cz: 'Šachy jsou jediná hra dost těžká na to, aby se to vyplatilo: jejich velká síť udrží dva workery skutečně vytížené, takže 2 je na GPU doporučené nastavení pro ~1,6× rychlejší self-play.',
      },
    },
  },

  az_learning_rate: {
    general: {
      en: '**Learning rate** — how big a step the network takes when it learns from each batch of self-play games.\nHigher learns faster but can wobble or overshoot; lower is steadier but slower. AlphaZero is sensitive here: too high and the network chases noisy early games and stalls, so this default is deliberately gentle.',
      cz: '**Rychlost učení** — jak velký krok síť udělá, když se učí z každé dávky self-play partií.\nVyšší se učí rychleji, ale může kolísat nebo přestřelit; nižší je stabilnější, ale pomalejší. AlphaZero je tu citlivý: příliš vysoká a síť se honí za zašuměnými ranými partiemi a uvázne, proto je výchozí hodnota záměrně mírná.',
    },
    recommended: {
      en: '5e-4 is a safe, steady default for self-play. There is rarely a reason to raise it much — gentle learning is what keeps AlphaZero stable.',
      cz: '5e-4 je bezpečná, stabilní výchozí hodnota pro self-play. Zřídka je důvod ji výrazně zvyšovat — mírné učení je to, co drží AlphaZero stabilní.',
    },
    range: '1e-4 – 3e-3',
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

  // The gold "Goal" line on the reward chart (G6c follow-up) — what "solved" means + why a curve can
  // sit far below it while still making real progress. Kept game-neutral: the key idea is that every
  // game is scored on its own scale, so the *gap* matters, not the raw number, and a very high ceiling
  // (the hardest tasks may never quite reach the line in a practical run) is normal, not a failure.
  goal: {
    general: {
      en: "**The gold line marks \"solved\"** — the score the agent is aiming for, the same target the skill meter calls 100%. The curve climbs toward it, so the gap between the curve and the line is how much there still is to learn.\nEvery game is scored on its own scale, so the line sits at a different height for each — what matters is the *gap*, not the raw number. A curve far below the line is normal, not broken: some tasks have a very high ceiling, and the hardest ones may never quite reach it in a practical run. Steady upward progress is the real sign of learning.",
      cz: '**Zlatá čára označuje „vyřešeno“** — skóre, ke kterému agent míří; je to stejný cíl, který měřič dovednosti označuje jako 100 %. Křivka k němu stoupá, takže mezera mezi křivkou a čárou ukazuje, kolik se ještě dá naučit.\nKaždá hra se boduje na vlastní škále, takže čára leží u každé jinak vysoko — důležitá je *mezera*, ne holé číslo. Křivka hluboko pod čárou je normální, ne chyba: některé úlohy mají velmi vysoký strop a ty nejtěžší ho v praktickém běhu nemusí nikdy úplně dosáhnout. Skutečnou známkou učení je vytrvalé stoupání.',
    },
    perEnv: {
      tictactoe: {
        en: "For board games the score is how the net does against the reference AI: **−1** = loses every game, **0** = draws every game, **+1** = wins every game. \"Solved\" is +1, but Tic-Tac-Toe is a forced draw with good play, so a well-trained net realistically plateaus near **0 (draws)** — climbing from −0.9 toward 0 is already mastery, even though it stays well below the +1 line.",
        cz: 'U deskových her je skóre to, jak si síť vede proti referenční AI: **−1** = prohraje každou partii, **0** = každou remizuje, **+1** = každou vyhraje. „Vyřešeno“ je +1, ale piškvorky jsou při dobré hře vynucená remíza, takže dobře natrénovaná síť reálně uvázne kolem **0 (remízy)** — posun z −0,9 k 0 už je mistrovství, i když zůstává hluboko pod čárou +1.',
      },
      connect_four: {
        en: "Score = how the net does vs the reference AI: **−1** loses all, **0** draws all, **+1** wins all (\"solved\"). A fresh net already scores around −0.6 (it wins a few by luck), then climbs as it learns. Beating the search AI *every* game (+1) is a very high bar, so expect the curve to head toward 0 and above — well below +1 is still real, hard-won skill.",
        cz: 'Skóre = jak si síť vede proti referenční AI: **−1** vše prohraje, **0** vše remizuje, **+1** vše vyhraje („vyřešeno“). Čerstvá síť má hned kolem −0,6 (pár partií vyhraje náhodou), pak stoupá, jak se učí. Porazit prohledávací AI v *každé* partii (+1) je velmi vysoká laťka, takže čekejte, že křivka zamíří k 0 a výš — i hluboko pod +1 je to skutečná, těžce nabytá dovednost.',
      },
      othello: {
        en: "Score = how the net does vs the reference (easy) AI: **−1** loses all, **0** draws all, **+1** wins all (\"solved\"). Othello is huge, so a fresh net starts low — around −0.7 — then climbs as it learns to beat the easy searcher, typically settling a bit above 0 (it wins more than it loses). Reaching +1 (winning *every* game) is a very high bar, so a curve well below the line is still genuine progress.",
        cz: 'Skóre = jak si síť vede proti referenční (lehké) AI: **−1** vše prohraje, **0** vše remizuje, **+1** vše vyhraje („vyřešeno“). Othello je obrovské, takže čerstvá síť začíná nízko — kolem −0,7 — a pak stoupá, jak se učí porážet lehkého prohledávače, obvykle se ustálí kousek nad 0 (vyhrává víc, než prohrává). Dosáhnout +1 (vyhrát *každou* partii) je velmi vysoká laťka, takže i křivka hluboko pod čárou je skutečný pokrok.',
      },
      breakthrough: {
        en: "Score = how the net does vs the reference (medium) AI: **−1** loses all, **0** draws all, **+1** wins all (\"solved\"). A fresh net starts deep in the red (around −0.9), then climbs as it learns to break through — it beats the *easy* searcher almost at once, so it is scored against the tougher *medium* one to keep the curve honest and rising rather than instantly pinned at the top. Settling above 0 means it wins more than it loses against a real searcher.",
        cz: 'Skóre = jak si síť vede proti referenční (střední) AI: **−1** vše prohraje, **0** vše remizuje, **+1** vše vyhraje („vyřešeno“). Čerstvá síť začíná hluboko v záporu (kolem −0,9) a pak stoupá, jak se učí prorážet — *lehkého* prohledávače porazí takřka okamžitě, takže se měří proti tvrdšímu *střednímu*, aby křivka zůstala poctivá a stoupala, místo aby se hned přilepila ke stropu. Ustálení nad 0 znamená, že proti skutečnému prohledávači vyhrává víc, než prohrává.',
      },
      chess: {
        en: "Score = how the net does vs a deliberately weak reference searcher: **−1** loses all, **0** draws all, **+1** wins all (\"solved\"). Chess is far too deep to beat a real engine on a short run, so the yardstick is a near-random searcher a learning net can actually catch — a fresh net already draws it (around 0, since both play randomly), then the curve climbs above 0 as self-play teaches it to win. Don't expect it near +1: this is an honest \"is it improving?\" signal, not a claim of strong chess.",
        cz: 'Skóre = jak si síť vede proti záměrně slabému referenčnímu prohledávači: **−1** vše prohraje, **0** vše remizuje, **+1** vše vyhraje („vyřešeno“). Šachy jsou příliš hluboké na to, aby v krátkém běhu porazily skutečný engine, takže měřítkem je téměř náhodný prohledávač, který učící se síť reálně dožene — čerstvá síť ho hned remizuje (kolem 0, protože oba hrají náhodně) a pak křivka stoupá nad 0, jak ji self-play učí vyhrávat. Nečekejte hodnoty u +1: je to poctivý signál „zlepšuje se?“, ne tvrzení o silné hře.',
      },
    },
  },

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
      hopper: {
        en: 'Hopper pays a little for each bit of forward hop and a small bonus for staying upright, and a fall ends the run, so the curve climbs from about +120 (just standing briefly) toward +3800 ("solved") as it learns to hop.',
        cz: 'Hopper platí trochu za každý kousek skoku vpřed a malý bonus za udržení vzpřímené polohy a pád běh ukončí, takže křivka stoupá zhruba z +120 (jen krátké stání) k +3800 („vyřešeno“), jak se učí skákat.',
      },
      walker2d: {
        en: 'Walker2d pays for forward progress plus a small upright bonus, and a fall ends the run, so the curve climbs from about +80 toward the thousands (+3500 is a strong gait) as it learns to walk.',
        cz: 'Walker2d platí za postup vpřed plus malý bonus za vzpřímenou polohu a pád běh ukončí, takže křivka stoupá zhruba z +80 k tisícům (+3500 je silná chůze), jak se učí chodit.',
      },
      halfcheetah: {
        en: 'HalfCheetah never falls, so the curve simply climbs from near 0 toward +4800 as the running gait gets faster; early flailing can dip it negative before it learns to move forward.',
        cz: 'HalfCheetah nikdy nepadá, takže křivka prostě stoupá od skoro 0 k +4800, jak se běh zrychluje; počáteční zmatené pohyby ji mohou stáhnout do záporu, než se naučí pohyb vpřed.',
      },
      ant: {
        en: 'Ant earns a per-step "healthy" bonus just for not flipping over, so the curve starts high (around +990 for standing still) and climbs toward +6000 as it learns to walk — that high baseline is normal for Ant, not a sign it has already learned.',
        cz: 'Ant získává bonus za „zdraví“ každý krok jen za to, že se nepřevrátí, takže křivka začíná vysoko (kolem +990 za stání na místě) a stoupá k +6000, jak se učí chodit — ta vysoká základna je u Antu normální, ne známka, že už se něco naučil.',
      },
      reacher: {
        en: 'Reacher pays the negative distance from the tip to the target each step (plus a little for effort), so the score is always negative; the curve climbs from about −12 (idle) toward −3.75 ("solved") as the arm learns to reach and hold.',
        cz: 'Reacher platí zápornou vzdálenost špičky od cíle každý krok (plus trochu za námahu), takže skóre je vždy záporné; křivka stoupá zhruba z −12 (nečinnost) k −3,75 („vyřešeno“), jak se rameno učí dosáhnout a udržet.',
      },
      swimmer: {
        en: 'Swimmer pays the forward speed minus a small effort cost, so the curve climbs from near 0 toward +360 as it finds a swimming rhythm; there is no fall, only the stroke to get right.',
        cz: 'Swimmer platí rychlost vpřed minus malou cenu za námahu, takže křivka stoupá od skoro 0 k +360, jak najde plavecký rytmus; není žádný pád, jen je třeba trefit záběr.',
      },
      humanoid: {
        en: 'Humanoid earns a per-step "healthy" bonus for staying upright, so the curve starts around +200 (just standing for a moment) and must climb toward +5000 as it learns to balance and step — that high baseline is normal, not a sign it has already learned. It is the hardest MuJoCo task, so expect a slow climb.',
        cz: 'Humanoid získává každý krok bonus za „zdraví“ za udržení vzpřímené polohy, takže křivka začíná kolem +200 (jen chvíli stojí) a musí stoupat k +5000, jak se učí balancovat a krokovat — ta vysoká základna je normální, ne známka, že už se něco naučil. Je to nejtěžší úloha MuJoCo, takže čekejte pomalé stoupání.',
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
      hopper: MUJOCO_LOSS,
      walker2d: MUJOCO_LOSS,
      halfcheetah: MUJOCO_LOSS,
      ant: MUJOCO_LOSS,
      reacher: MUJOCO_LOSS,
      swimmer: MUJOCO_LOSS,
      humanoid: MUJOCO_LOSS,
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
