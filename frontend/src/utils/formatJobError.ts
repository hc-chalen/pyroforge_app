/** Turn raw Celery / HTTP errors into user-facing messages. */
export function formatJobError(error: string | undefined, fallback: string): string {
  if (!error?.trim()) return fallback;

  const message = error.trim();
  const lowered = message.toLowerCase();

  // Whole numbers only: the "504" inside "length 5041" is not a gateway timeout.
  if (/\b504\b/.test(message) || lowered.includes("gateway timeout")) {
    return (
      "The structure prediction service timed out. This is a server-side " +
      "problem, not your sequence — retry in a few minutes."
    );
  }
  if (/\b422\b/.test(message) || lowered.includes("rejected this sequence")) {
    return (
      "ESMFold rejected this sequence. Use a valid protein sequence " +
      "between 10 and 400 residues."
    );
  }
  if (lowered.includes("exceeds the esmfold limit")) return message;
  if (lowered.includes("invalid amino acid")) return message;
  if (lowered.includes("timed out")) return message;
  if (lowered.includes("could not reach the esmfold")) return message;
  if (lowered.includes("esmfold api error")) return message;
  if (lowered.includes("httperror") || lowered.includes("httpstatuserror")) {
    return (
      "Structure folding failed due to an upstream API error. " +
      "Please retry in a few minutes."
    );
  }

  return message;
}
