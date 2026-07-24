import {
  createTheme,
  type CSSVariablesResolver,
  type MantineColorsTuple,
} from "@mantine/core";

const brand: MantineColorsTuple = [
  "#e8f1f9",
  "#c8dcf0",
  "#96c0e4",
  "#6fa3d8",
  "#598abd",
  "#42719f",
  "#2c5985",
  "#254b70",
  "#1e3d5b",
  "#172f47",
];

const surface: MantineColorsTuple = [
  "#ffffff",
  "#f4f6f8",
  "#eef1f5",
  "#e4e9ef",
  "#dde3e9",
  "#cfd6de",
  "#b7c0ca",
  "#98a3af",
  "#79838f",
  "#657283",
];

const darkSurface: MantineColorsTuple = [
  "#e7eaed",
  "#c5ccd3",
  "#a3adb6",
  "#8f9aa5",
  "#2a323c",
  "#232a33",
  "#1d242c",
  "#161c23",
  "#0f1318",
  "#0a0d11",
];

const flatStyles = {
  cardBorder: "1px solid var(--mantine-color-default-border)",
};

const inputStyles = {
  fontFamily: "'IBM Plex Mono', monospace",
  boxShadow: "none",
  border: "1px solid var(--mantine-color-default-border)",
  backgroundColor: "var(--pyro-input-bg)",
  color: "var(--mantine-color-text)",
};

export const cssVariablesResolver: CSSVariablesResolver = () => ({
  variables: {},
  light: {
    "--mantine-color-body": "#f4f6f8",
    "--mantine-color-text": "#1b2530",
    "--mantine-color-dimmed": "#657283",
    "--mantine-color-default": "#ffffff",
    "--mantine-color-default-hover": "#eef1f5",
    "--mantine-color-default-color": "#1b2530",
    "--mantine-color-default-border": "#dde3e9",
    "--pyro-shadow-paper": "none",
    "--pyro-shadow-inset": "none",
    "--pyro-shadow-shell": "none",
    "--pyro-shadow-header": "none",
    "--pyro-item-border": "1px solid #dde3e9",
    "--pyro-item-bg": "#ffffff",
    "--pyro-item-selected-bg": "#eef1f5",
    "--pyro-card-bg": "#ffffff",
    "--pyro-card-nested-bg": "#eef1f5",
    "--pyro-input-bg": "#eef1f5",
    "--pyro-viewer-bg": "#e7ebef",
  },
  dark: {
    "--mantine-color-body": "#0f1318",
    "--mantine-color-text": "#e7eaed",
    "--mantine-color-dimmed": "#8f9aa5",
    "--mantine-color-default": "#161c23",
    "--mantine-color-default-hover": "#1d242c",
    "--mantine-color-default-color": "#e7eaed",
    "--mantine-color-default-border": "#2a323c",
    "--pyro-shadow-paper": "none",
    "--pyro-shadow-inset": "none",
    "--pyro-shadow-shell": "none",
    "--pyro-shadow-header": "none",
    "--pyro-item-border": "1px solid #2a323c",
    "--pyro-item-bg": "#161c23",
    "--pyro-item-selected-bg": "#1d242c",
    "--pyro-card-bg": "#161c23",
    "--pyro-card-nested-bg": "#1d242c",
    "--pyro-input-bg": "#1d242c",
    "--pyro-viewer-bg": "#1a2029",
  },
});

export const theme = createTheme({
  primaryColor: "brand",
  primaryShade: { light: 6, dark: 3 },
  autoContrast: true,
  colors: {
    brand,
    surface,
    dark: darkSurface,
  },
  black: "#1b2530",
  fontFamily: "'IBM Plex Sans', sans-serif",
  fontFamilyMonospace: "'IBM Plex Mono', monospace",
  headings: {
    fontFamily: "'IBM Plex Sans', sans-serif",
    fontWeight: "600",
  },
  defaultRadius: "sm",
  spacing: {
    xs: "0.5rem",
    sm: "0.625rem",
    md: "0.875rem",
    lg: "1.25rem",
    xl: "1.5rem",
  },
  radius: {
    xs: "4px",
    sm: "6px",
    md: "8px",
    lg: "10px",
    xl: "14px",
  },
  components: {
    Paper: {
      defaultProps: {
        radius: "lg",
        p: "lg",
      },
      styles: {
        root: {
          backgroundColor: "var(--pyro-card-bg)",
          color: "var(--mantine-color-text)",
          border: flatStyles.cardBorder,
          boxShadow: "none",
        },
      },
    },
    Title: {
      styles: {
        root: {
          color: "var(--mantine-color-text)",
        },
      },
    },
    Table: {
      styles: {
        table: {
          color: "var(--mantine-color-text)",
        },
        th: {
          color: "var(--mantine-color-text)",
        },
        td: {
          color: "var(--mantine-color-text)",
        },
      },
    },
    Textarea: {
      styles: {
        input: inputStyles,
        label: {
          color: "var(--mantine-color-text)",
        },
      },
    },
    TextInput: {
      styles: {
        input: {
          ...inputStyles,
          fontFamily: "'IBM Plex Sans', sans-serif",
        },
        label: {
          color: "var(--mantine-color-text)",
        },
      },
    },
    NumberInput: {
      styles: {
        input: {
          ...inputStyles,
        },
        label: {
          color: "var(--mantine-color-text)",
        },
      },
    },
    Select: {
      styles: {
        input: {
          ...inputStyles,
          fontFamily: "'IBM Plex Sans', sans-serif",
        },
        label: {
          color: "var(--mantine-color-text)",
        },
        option: {
          color: "var(--mantine-color-text)",
        },
      },
    },
    Modal: {
      styles: {
        content: {
          backgroundColor: "var(--pyro-card-bg)",
          color: "var(--mantine-color-text)",
        },
        header: {
          backgroundColor: "var(--pyro-card-bg)",
        },
        title: {
          color: "var(--mantine-color-text)",
          fontWeight: 700,
        },
      },
    },
    AppShell: {
      styles: {
        navbar: {
          backgroundColor: "var(--pyro-card-bg)",
          borderRight: flatStyles.cardBorder,
          color: "var(--mantine-color-text)",
        },
        header: {
          backgroundColor: "var(--pyro-card-bg)",
          borderBottom: flatStyles.cardBorder,
          color: "var(--mantine-color-text)",
        },
        main: {
          backgroundColor: "var(--mantine-color-body)",
          color: "var(--mantine-color-text)",
        },
      },
    },
  },
});
