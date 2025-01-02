# Log File Explanation

## bridge.log

This is the log of the output you see at the console when you launch the worker.

## bridge_0.log

This is the output of the safety process. This process usually does not experience issues. There typically is not any useful information for debugging.

## bridge_1.log, bridge_2.log, etc

This is the output of the processes which actually do the generations. They correspond to the main console log output (and therefore bridge.log) as "Process 1" (being bridge_1.log) and so forth.

## trace.log

This has only the errors that appeared in the main console output. You can find these in context by reviewing bridge.log.

## trace_0.log, trace_1.log, trace_2.log, etc

These have only the errors that appeared in the working processes. You can find these in context by reviewing the corresponding bridge_n.log.

## stderr.log, stdout.log

These are logs of last resort and are typically only populated when something went wrong. There may occasionally be information in these, but they are often empty.
