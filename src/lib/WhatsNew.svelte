<!-- SPDX-License-Identifier: GPL-3.0-only -->
<!-- Copyright (C) 2026 NeuroSkill.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 only. -->
<!--
  WhatsNew — modal dialog shown once per app version.

  Shown automatically on startup when the running version differs from the
  version stored in localStorage under the key "whatsNew.lastSeenVersion".
  Dismissing the dialog (via the button or the ✕) persists the current
  version so it is never shown again for that release.
-->
<script lang="ts">
  import { onMount }           from "svelte";
  import { invoke }            from "@tauri-apps/api/core";
  import * as Dialog           from "$lib/components/ui/dialog";
  import { Button }            from "$lib/components/ui/button";
  import { t }                 from "$lib/i18n/index.svelte";
  import changelogRaw          from "../../CHANGELOG.md?raw";

  // ── localStorage key ───────────────────────────────────────────────────────
  const STORAGE_KEY = "whatsNew.lastSeenVersion";

  // ── State ──────────────────────────────────────────────────────────────────
  let open        = $state(false);
  let appVersion  = $state("…");

  // ── Changelog parsing ──────────────────────────────────────────────────────

  interface ChangeSection {
    heading: string;   // ### heading text (empty string = no heading, top-level items)
    items:   string[]; // bullet-point lines (markdown stripped)
  }

  interface VersionEntry {
    version:  string;
    date:     string;
    sections: ChangeSection[];
  }

  /**
   * Parse the raw CHANGELOG.md into structured version entries.
   *
   * Expected format:
   *   ## [0.0.6] — 2026-03-06
   *   ### Section heading
   *   - bullet item
   *   - another item
   */
  function parseChangelog(raw: string): VersionEntry[] {
    const entries: VersionEntry[] = [];

    // Split on lines that start a top-level version block
    const versionBlockRe = /^##\s+\[([^\]]+)\]\s*[—–-]+\s*(\S+)/m;
    const blocks = raw.split(/^(?=##\s+\[)/m).filter(b => b.trim());

    for (const block of blocks) {
      const headerMatch = block.match(versionBlockRe);
      if (!headerMatch) continue;

      const version = headerMatch[1].trim();
      const date    = headerMatch[2].trim();
      const body    = block.slice(block.indexOf("\n") + 1);

      const sections: ChangeSection[] = [];
      let   current:  ChangeSection   = { heading: "", items: [] };

      for (const rawLine of body.split("\n")) {
        const line = rawLine.trimEnd();

        if (/^###\s/.test(line)) {
          // New sub-section — flush the previous one if it has items
          if (current.items.length > 0 || current.heading) sections.push(current);
          current = { heading: line.replace(/^###\s+/, "").trim(), items: [] };
        } else if (/^[-*+]\s/.test(line)) {
          // Bullet item — strip leading marker
          current.items.push(line.replace(/^[-*+]\s+/, "").trim());
        }
        // Skip blank lines, horizontal rules, comments, etc.
      }

      // Flush the last section
      if (current.items.length > 0 || current.heading) sections.push(current);
      if (sections.length > 0) entries.push({ version, date, sections });
    }

    return entries;
  }

  const changelog: VersionEntry[] = parseChangelog(changelogRaw);

  /** The latest entry is whatever comes first in the changelog. */
  const latest: VersionEntry | undefined = changelog[0];

  // ── Lifecycle ──────────────────────────────────────────────────────────────
  onMount(async () => {
    try {
      appVersion = await invoke<string>("get_app_version");
    } catch {
      // Fall back to package.json version embedded at build time
      appVersion = latest?.version ?? "?";
    }

    // Show the dialog only when this version hasn't been seen yet
    try {
      const seen = localStorage.getItem(STORAGE_KEY);
      if (seen !== appVersion) open = true;
    } catch { /* private-browsing / SSR */ }
  });

  // ── Dismiss ────────────────────────────────────────────────────────────────
  function dismiss() {
    open = false;
    try { localStorage.setItem(STORAGE_KEY, appVersion); } catch {}
  }

  // Intercept the dialog's own close button (✕) as well
  function handleOpenChange(next: boolean) {
    if (!next) dismiss();
    else open = true;
  }
</script>

{#if latest}
  <Dialog.Root bind:open onOpenChange={handleOpenChange}>
    <Dialog.Content
      class="max-w-lg w-full p-0 overflow-hidden border border-border dark:border-white/[0.08]
             bg-background rounded-2xl gap-0"
    >

      <!-- ── Gradient header ──────────────────────────────────────────────── -->
      <div class="relative px-6 pt-6 pb-5
                  bg-gradient-to-br from-violet-500/10 via-blue-500/8 to-sky-500/10
                  dark:from-violet-500/15 dark:via-blue-500/12 dark:to-sky-500/15
                  border-b border-border dark:border-white/[0.06]">

        <!-- sparkle icon -->
        <div class="flex items-center gap-3 mb-3">
          <div class="flex items-center justify-center w-10 h-10 rounded-xl shrink-0
                      bg-gradient-to-br from-violet-500 to-blue-600
                      shadow-lg shadow-violet-500/30 dark:shadow-violet-500/40">
            <span class="text-lg leading-none select-none" aria-hidden>✨</span>
          </div>
          <div class="flex flex-col gap-0.5">
            <span class="text-[0.9rem] font-bold leading-tight text-foreground">
              {t("whatsNew.title")}
            </span>
            <span class="text-[0.6rem] font-semibold text-muted-foreground/60 tracking-wide uppercase">
              {t("whatsNew.version", { version: latest.version })}
              &nbsp;·&nbsp;
              {latest.date}
            </span>
          </div>
        </div>

      </div>

      <!-- ── Scrollable changelog body ───────────────────────────────────── -->
      <div class="px-6 py-5 max-h-[55vh] overflow-y-auto overscroll-contain
                  flex flex-col gap-5 text-[0.78rem]">

        {#each latest.sections as section}
          <div class="flex flex-col gap-2">

            {#if section.heading}
              <h3 class="text-[0.72rem] font-bold tracking-wide uppercase
                         text-violet-600 dark:text-violet-400 leading-tight">
                {section.heading}
              </h3>
            {/if}

            <ul class="flex flex-col gap-1.5 pl-0">
              {#each section.items as item}
                <li class="flex items-start gap-2 text-[0.75rem] text-foreground/85 leading-relaxed">
                  <span class="mt-[0.35em] shrink-0 w-1.5 h-1.5 rounded-full
                               bg-violet-400/70 dark:bg-violet-500/70"></span>
                  <span>{item}</span>
                </li>
              {/each}
            </ul>

          </div>
        {/each}

      </div>

      <!-- ── Footer ──────────────────────────────────────────────────────── -->
      <div class="px-6 py-4 border-t border-border dark:border-white/[0.06]
                  flex items-center justify-end">
        <Button
          size="sm"
          class="px-6 h-9 text-[0.78rem] font-semibold
                 bg-gradient-to-r from-violet-500 to-blue-600
                 hover:from-violet-600 hover:to-blue-700
                 text-white shadow shadow-violet-500/20 dark:shadow-violet-500/30
                 border-0"
          onclick={dismiss}
        >
          {t("whatsNew.gotIt")}
        </Button>
      </div>

    </Dialog.Content>
  </Dialog.Root>
{/if}
