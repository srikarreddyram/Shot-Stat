import { createFileRoute, redirect } from "@tanstack/react-router";

// The model's features now live as a tab of /model, beside the shot
// archetypes. Kept as a redirect so existing links and bookmarks still work.
export const Route = createFileRoute("/model-features")({
  beforeLoad: () => {
    throw redirect({ to: "/model", hash: "features" });
  },
});
