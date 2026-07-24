import { Alert, Button, Stack, Text } from "@mantine/core";
import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  title?: string;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("ErrorBoundary caught:", error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <Alert color="red" title={this.props.title ?? "Something went wrong"}>
          <Stack gap="sm">
            <Text size="sm">{this.state.error.message}</Text>
            <Button
              size="xs"
              variant="light"
              onClick={() => this.setState({ error: null })}
            >
              Try again
            </Button>
          </Stack>
        </Alert>
      );
    }
    return this.props.children;
  }
}
