import { createFileRoute, redirect } from "@tanstack/react-router";

// Shot archetypes now live as a tab of /model, beside the model's features.
// Kept as a redirect so existing links and bookmarks still land on them.
export const Route = createFileRoute("/archetypes")({
  beforeLoad: () => {
    throw redirect({ to: "/model", hash: "archetypes" });
  },
});
