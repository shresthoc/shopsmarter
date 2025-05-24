const LOG_LEVELS = {
  DEBUG: 'DEBUG',
  INFO: 'INFO',
  WARN: 'WARN',
  ERROR: 'ERROR'
} as const;

type LogLevel = typeof LOG_LEVELS[keyof typeof LOG_LEVELS];

class Logger {
  private static instance: Logger;
  private isDevelopment: boolean;

  private constructor() {
    this.isDevelopment = process.env.NODE_ENV === 'development';
  }

  static getInstance(): Logger {
    if (!Logger.instance) {
      Logger.instance = new Logger();
    }
    return Logger.instance;
  }

  private formatMessage(level: LogLevel, message: string, data?: any): string {
    const timestamp = new Date().toISOString();
    const dataString = data ? `\nData: ${JSON.stringify(data, null, 2)}` : '';
    return `[${timestamp}] [${level}] ${message}${dataString}`;
  }

  debug(message: string, data?: any): void {
    if (this.isDevelopment) {
      console.debug(this.formatMessage(LOG_LEVELS.DEBUG, message, data));
    }
  }

  info(message: string, data?: any): void {
    console.info(this.formatMessage(LOG_LEVELS.INFO, message, data));
  }

  warn(message: string, data?: any): void {
    console.warn(this.formatMessage(LOG_LEVELS.WARN, message, data));
  }

  error(message: string, error?: Error | any): void {
    console.error(this.formatMessage(LOG_LEVELS.ERROR, message, {
      error: error instanceof Error ? {
        message: error.message,
        stack: error.stack,
        name: error.name
      } : error
    }));
  }
}

export const logger = Logger.getInstance(); 