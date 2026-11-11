import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { createStepProvider, withChineseReasoning } from "./step-teacher.js";

export default function stepExtension(pi: ExtensionAPI): void {
  pi.registerProvider(createStepProvider());
  pi.on("input", async (event, ctx) => {
    if (ctx.model?.provider !== "stepfun") return { action: "continue" as const };
    return { action: "transform" as const, text: withChineseReasoning(event.text), images: event.images };
  });
}
